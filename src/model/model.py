from numpy import NaN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


class baseModule(nn.Module):
    def __init__(self):
        super(baseModule, self).__init__()

    def loss_function(self, p, q, reduce="mean"):
        logits = p
        labels = q
        loss = -torch.sum(logits*labels, dim=2)
        loss = torch.mean(loss).squeeze()
        return loss

    def reset_parameter(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_normal_(m.weight)


################## Edge GNN ########################

class EdgeGNN_layer(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, message_dim):
        super(EdgeGNN_layer, self).__init__()
        self.message_edge_lin = nn.Sequential(nn.Linear(edge_feat_dim+message_dim, message_dim, bias=True), nn.ReLU())
        self.message_pass_lin = nn.Sequential(nn.Linear(message_dim, message_dim), nn.ReLU())
        self.message_node_lin = nn.Sequential(nn.Linear(node_feat_dim+message_dim, message_dim), nn.ReLU())
        self.gates = nn.GRUCell(message_dim, message_dim, bias=True)

    def forward(self, inputs):
        _, _, edge_feat, message_old, nb_adj = inputs
        message = self.message_edge_lin(torch.cat((message_old, edge_feat), dim=1))
        message = torch.sparse.mm(nb_adj, message)
        message = self.message_pass_lin(message)
        # gates
        message = self.gates(message, message_old)
        return message

class EdgeGNN(baseModule):
    def __init__(self, num_status, node_feat_dim, edge_feat_dim, message_dim, number_layers, device):
        super(EdgeGNN, self).__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.message_dim = message_dim
        self.number_layers = number_layers
        self.num_status = num_status
        self.device = device

        self.node_emb = nn.Sequential(nn.Linear(2, node_feat_dim, bias=True), nn.ReLU())
        self.edge_emb = nn.Sequential(nn.Linear(1, edge_feat_dim, bias=True), nn.ReLU())
        self.init_lin = nn.Sequential(nn.Linear(node_feat_dim+edge_feat_dim, message_dim, bias=True), nn.ReLU())
        self.aggr_lin = nn.Sequential(nn.Linear(message_dim, message_dim, bias=True), nn.ReLU())
        self.node_lin = nn.Sequential(nn.Linear(message_dim+node_feat_dim, self.num_status, bias=True), nn.ReLU())

        self.message_passing = EdgeGNN_layer(self.node_feat_dim, self.edge_feat_dim, self.message_dim)
        self.reset_parameter()

        self.to(self.device)

    def forward(self, inputs):
        snode2edge, tnode2edge, edge2tnode, nb_matrix, edge_feat, node_prob, node_seed = inputs

        # embedding node and edge feature
        node_feat_ori = self.node_emb(torch.cat((node_prob, node_seed), dim=1))
        snode_feat = torch.sparse.mm(snode2edge, node_feat_ori)
        tnode_feat = torch.sparse.mm(tnode2edge, node_feat_ori)
        edge_feat = self.edge_emb(edge_feat) # [E, F]

        # initial values for message
        message = torch.cat((snode_feat, edge_feat), dim=1)
        message = self.init_lin(message)
        
        # message passing
        message_delta = []
        marginal_log = []

        # readout
        marginal_log.append(self.nodes_out(edge2tnode, message, node_feat_ori)) # start from t=1

        for _ in range(self.number_layers):
            message = self.message_passing([snode_feat, tnode_feat, edge_feat, message, nb_matrix])
            marginal_log.append(self.nodes_out(edge2tnode, message, node_feat_ori))
            delta = self.marginal_delta(marginal_log)
            message_delta.append(delta)
            if delta < 5E-5:
                break

        marginal = torch.stack(marginal_log, dim=0)
        return marginal, message_delta

    def nodes_out(self, edge2tnode, message, node_feat_seed_ori):
        node_agg = torch.sparse.mm(edge2tnode, message)
        node_agg = self.aggr_lin(node_agg)  
        node_agg = torch.cat((node_agg, node_feat_seed_ori), dim=1)
        marginal = F.log_softmax(self.node_lin(node_agg), dim=1)
        return marginal
    
    def marginal_delta(self, marginal_log):
        marginal_new = torch.exp(marginal_log[-2][:, -1])
        marginal_old = torch.exp(marginal_log[-1][:, -1])

        return torch.max(torch.abs(marginal_new-marginal_old)).item()


################## NEDMP ########################
class NEDMP_layer(nn.Module):
    def __init__(self, hid_dim):
        super(NEDMP_layer, self).__init__()
        self.theta_emb= nn.Sequential(nn.Linear(3, hid_dim), nn.ReLU())
        self.message_emb1 = nn.Sequential(nn.Linear(1, hid_dim), nn.ReLU())
        self.message_emb2 = nn.Sequential(nn.Linear(1, hid_dim), nn.ReLU())
        self.agg_layer1 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU())
        self.agg_layer2 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU())
        self.cat_hidden_layer = nn.Sequential(nn.Linear(hid_dim*2, hid_dim), nn.ReLU())
        self.gates = nn.GRUCell(hid_dim, hid_dim, bias=True)

        self.scale_delta1 = nn.Sequential(nn.Linear(2*hid_dim, 2),
                                          nn.Sigmoid())
        self.scale_delta2 = nn.Sequential(nn.Linear(2*hid_dim, 2),
                                          nn.Sigmoid())
    def init(self, theta):
        self.hidden = self.theta_emb(theta)
    
    def forward(self, theta, edge_message, node_message, nb_matrix, edge2tnode):
        theta_emb = self.theta_emb(theta)
        node_message = self.message_emb1(node_message.reshape(-1, 1))
        edge_message = self.message_emb2(edge_message.reshape(-1, 1))

        hidden = self.cat_hidden_layer(torch.cat((self.hidden, theta_emb), dim=1))

        node_agg = self.agg_layer1(torch.mm(edge2tnode, hidden))
        cat_emb = torch.cat((node_agg, node_message), dim=1)
        node_res = self.scale_delta1(cat_emb)
        node_scale, node_delta = node_res[:, 0], node_res[:, 1]

        hidden_agg = self.agg_layer2(torch.mm(nb_matrix, hidden))
        self.hidden = self.gates(hidden_agg, self.hidden)

        # cat_emb = torch.cat((self.hidden, edge_message), dim=1)
        cat_emb = torch.cat((hidden_agg, edge_message), dim=1)
        edge_res = self.scale_delta2(cat_emb)
        edge_scale, edge_delta = edge_res[:, 0], edge_res[:, 1]

        
        return node_scale, node_delta, edge_scale, edge_delta

class NEDMP(baseModule):
    def __init__(self, hid_dim, number_layers, device):
        super(NEDMP, self).__init__()
        self.number_layers = number_layers
        self.device = device
        self.mpnn = NEDMP_layer(hid_dim).to(device)

    def forward(self, inputs):
        edge2tnode, nb_matrix, adj_index, cave_index, weights, nodes_gamma, seed_list = inputs

        self.marginals = []
        self.regulars = []
        message_delta = []

        self.edge2tnode = edge2tnode
        self.nb_matrix = nb_matrix
        self.cave_index = cave_index
        self.edge_index = adj_index
        self.src_nodes = adj_index[0]
        self.tar_nodes = adj_index[1]
        self.weights = weights
        self.nodes_gamma = nodes_gamma
        self.gamma = nodes_gamma[self.src_nodes]
        self.N = max([torch.max(self.src_nodes), torch.max(self.tar_nodes)]).item()+1
        self.E = len(self.src_nodes)

        # init
        self.seeds = torch.zeros(self.N).to(self.device)
        self.seeds[seed_list] = 1

        self.Ps_0 = 1 - self.seeds
        self.Pi_0 = self.seeds
        self.Pr_0 = torch.zeros_like(self.seeds).to(self.device)

        self.Ps_i_0 = self.Ps_0[self.src_nodes]
        self.Pi_i_0 = self.Pi_0[self.src_nodes]
        self.Pr_i_0 = self.Pr_0[self.src_nodes]

        self.Phi_ij_0 = 1 - self.Ps_i_0
        self.Theta_ij_0 = torch.ones(self.E).to(self.device)    
        
        # first iteration, t = 1
        self.Theta_ij_t = self.Theta_ij_0 - self.weights * self.Phi_ij_0 + 1E-20 # get rid of NaN
        self.Ps_ij_t_1 = self.Ps_i_0 # t-1
        self.Ps_ij_t = self.Ps_i_0 * self.mulmul(self.Theta_ij_t) # t
        self.Phi_ij_t = (1-self.weights)*(1-self.gamma)*self.Phi_ij_0 - (self.Ps_ij_t-self.Ps_ij_t_1)

        # marginals (t=1)
        self.Ps_t = self.Ps_0 * scatter(self.Theta_ij_t, self.tar_nodes, reduce="mul", dim_size=self.N)
        self.Pr_t = self.Pr_0 + self.nodes_gamma*self.Pi_0
        self.Pi_t = 1 - self.Ps_t - self.Pr_t
        self.marginals.append(torch.stack([self.Ps_t, self.Pi_t, self.Pr_t], dim=1))
        
        message = torch.stack((self.Theta_ij_t, self.Phi_ij_t, self.Ps_ij_t), dim=1)
        self.mpnn.init(message)
        # Iteration
        for l in range(self.number_layers):
            self.iteration()
            delta = self.marginal_delta(self.marginals)
            message_delta.append(delta)
            if delta < 5E-5:
                break
 
        marginals = torch.stack(self.marginals, dim=0) # [T, N, 3]

        self.regular_loss()
        
        # TODO: 合理吗？    
        marginals[marginals<=0] = 1E-20
        marginals[marginals>1] = 1

        marginals = torch.log(marginals)

        return marginals, message_delta

    
    def iteration(self):
        self.Theta_ij_t = self.Theta_ij_t - self.weights * self.Phi_ij_t

        edge_message = self.mulmul(self.Theta_ij_t)
        node_messgae = scatter(self.Theta_ij_t, self.tar_nodes, reduce="mul", dim_size=self.N)

        message = torch.stack((self.Theta_ij_t, self.Phi_ij_t, self.Ps_ij_t), dim=1)
        node_scale, node_delta, edge_scale, edge_delta = self.mpnn(message, edge_message, node_messgae, self.nb_matrix, self.edge2tnode)
        edge_message = edge_message * edge_scale + edge_delta
        node_messgae = node_messgae * node_scale + node_delta

        # Read out node-wise prediction
        node_messgae[node_messgae>1] = 1 
        Ps_t = self.Ps_0 * node_messgae
        ###
        self.regulars.append((Ps_t-self.Ps_t)[Ps_t>self.Ps_t])
        self.Ps_t = torch.where(Ps_t>self.Ps_t, self.Ps_t, Ps_t)
        ###
        Pr_t = self.Pr_t + self.nodes_gamma*self.Pi_t
        self.regulars.append((self.Pr_t-Pr_t)[Pr_t<self.Pr_t])

        self.Pr_t = torch.where(Pr_t<self.Pr_t, self.Pr_t, Pr_t)
        self.Pi_t = 1 - self.Ps_t - self.Pr_t
        self.marginals.append(torch.stack([self.Ps_t, self.Pi_t, self.Pr_t], dim=1))

        # Iteration
        edge_message[edge_message>1] = 1
        new_Ps_ij_t = self.Ps_i_0 * edge_message
        Ps_ij_t_1 = self.Ps_ij_t
        self.Ps_ij_t = new_Ps_ij_t
        self.Phi_ij_t = (1-self.weights)*(1-self.gamma)*self.Phi_ij_t - (self.Ps_ij_t-Ps_ij_t_1)


    def mulmul(self, Theta_t):
        Theta = scatter(Theta_t, index=self.tar_nodes, reduce="mul", dim_size=self.N) # [N]
        Theta = Theta[self.src_nodes] #[E]
        Theta_cav = scatter(Theta_t, index=self.cave_index, reduce="mul", dim_size=self.E+1)[:self.E]

        mul = Theta / Theta_cav
        return mul

    def marginal_delta(self, marginal_log):
        if len(marginal_log)<=1:
            return 1000
        else:
            marginal_new = marginal_log[-1]
            marginal_old = marginal_log[-2]
            return torch.max(torch.abs(marginal_new-marginal_old)).item()
    
    def regular_loss(self):
        loss = torch.mean(torch.cat(self.regulars))
        self.regularLoss = 0 if torch.isnan(loss) else loss

    def loss_function(self, p, q):
        logits = p
        labels = q
        loss = -torch.sum(logits*labels, dim=2)
        loss = torch.mean(loss).squeeze()
        
        return loss + 10*self.regularLoss


class NEDMPR(baseModule):
    def __init__(self, hid_dim, number_layers, device):
        super(NEDMP, self).__init__()
        self.number_layers = number_layers
        self.device = device
        self.mpnn = NEDMP_layer(hid_dim).to(device)

    def forward(self, inputs):
        edge2tnode, nb_matrix, adj_index, cave_index, weights, nodes_gamma, seed_list = inputs

        self.marginals = []
        message_delta = []

        self.edge2tnode = edge2tnode
        self.nb_matrix = nb_matrix
        self.cave_index = cave_index
        self.edge_index = adj_index
        self.src_nodes = adj_index[0]
        self.tar_nodes = adj_index[1]
        self.weights = weights
        self.nodes_gamma = nodes_gamma
        self.gamma = nodes_gamma[self.src_nodes]
        self.N = max([torch.max(self.src_nodes), torch.max(self.tar_nodes)]).item()+1
        self.E = len(self.src_nodes)

        # init
        self.seeds = torch.zeros(self.N).to(self.device)
        self.seeds[seed_list] = 1

        self.Ps = 1 - self.seeds
        self.Pi = self.seeds

        self.Pij = torch.zeros(self.E) # that i is in the Infectious state as a result of being infected from one of its neighbors other than j
        for i, src in enumerate(self.src_nodes):
            if src in seed_list:
                self.Pij[i] = 1

        self.max_steps = 30
        for i in range(self.max_steps):
            self.iteration()


    
    def iteration(self):
        message = NaN
        message_aggregation = scatter(self.Pij * self.weights * self.Ps[self.tar_ndoes], self.tar_ndoes, reduce="sum")
    
        # message update
        message_aggregation_cave = message_aggregation[self.src_nodes] - message[self.cave_idx]
        self.message = self.message - self.node_prob_edge * self.message + message_aggregation_cave

        # Nodes update
        self.I = self.I - self.node_prob * self.I + message_aggregation
        self.S = 1 - self.I

        return self.I, self.S

    def _set_seeds(self, seed_list):
        """
        setting the initial conditions using seed_list
        """
        # The probabilities being infectious and susceptible
        self.I = torch.zeros(self.number_of_nodes)
        self.S = torch.ones(self.number_of_nodes)
        for seed in seed_list:
            self.I[seed] = 1
            self.S[seed] = 0
        self.record()
        # self.message[i] is the message for edge [src_node[i], tar_node[i]]
        # If src_node[i] is seed node, then self.message[i] = 1, else 0
        self.message = torch.zeros(self.number_of_edges)
        for i, src in enumerate(self.src_nodes):
            if src in seed_list:
                self.message[i] = 1
    
    def record(self):
        """
        recording a [N, 2] tensor for each step
        """
        I = deepcopy(self.I)
        S = deepcopy(self.S)
        self.marginal_each_step.append(torch.stack((S, I), dim=1))



    def _stop(self):
        if len(self.marginal_each_step) < 2:
            return False
        else:
            former, later = self.marginal_each_step[-2:]
            delta = torch.max(torch.abs(former-later))
            if delta > 0.0001:
                return False
            else:
                return True
    
    def run(self, seed_list):
        assert isinstance(seed_list, list)
        seed_list = [int(seed) for seed in seed_list]
        self._set_seeds(seed_list)
        for step in range(self.max_steps):
            self.iteration()
            self.record()
            if self._stop():
                break
        # stack marginals for output
        marginals = torch.stack(self.marginal_each_step, dim=0) # ==> [T, N, 2]
        return marginals


################## NodeGNN ########################
class NodeGNN_layer(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, message_dim):
        super(NodeGNN_layer, self).__init__()
        self.message_edge_lin = nn.Sequential(nn.Linear(edge_feat_dim+message_dim, message_dim, bias=True), nn.ReLU())
        self.message_pass_lin = nn.Sequential(nn.Linear(message_dim, message_dim), nn.ReLU())
        self.gates = nn.GRUCell(message_dim, message_dim, bias=True)

    def forward(self, inputs):
        edge_feat, snode2edge, edge2tnode, message_old = inputs
        message = torch.mm(snode2edge, message_old)
        message = self.message_edge_lin(torch.cat((message, edge_feat), dim=1))
        message = self.message_pass_lin(torch.mm(edge2tnode, message))
        message = self.gates(message, message_old)
        return message

class NodeGNN(baseModule):
    def __init__(self, num_status, node_feat_dim, edge_feat_dim, message_dim, number_layers, device):
        super(NodeGNN, self).__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.message_dim = message_dim
        self.number_layers = number_layers
        self.device = device
        self.num_status = num_status

        self.node_emb = nn.Sequential(nn.Linear(2, node_feat_dim, bias=True), nn.ReLU())
        self.edge_emb = nn.Sequential(nn.Linear(1, edge_feat_dim, bias=True), nn.ReLU())
        self.init_lin = nn.Sequential(nn.Linear(node_feat_dim, message_dim, bias=True), nn.ReLU())
        self.node_lin = nn.Linear(message_dim, num_status, bias=True)
        self.message_passing = NodeGNN_layer(self.node_feat_dim, self.edge_feat_dim, self.message_dim)

        self.reset_parameter()
        self.to(self.device)

    def forward(self, inputs):
        snode2edge, _, edge2tnode, _, edge_feat, node_prob, node_seed = inputs
        # embedding node and edge feature
        node_feat = self.node_emb(torch.cat((node_prob, node_seed), dim=1))
        edge_feat = self.edge_emb(edge_feat)
        # initial values for message
        message = self.init_lin(node_feat)
        
        # message passing
        message_delta = []
        marginal_log = []
        for _ in range(self.number_layers):
            message = self.message_passing([edge_feat, snode2edge, edge2tnode, message])
            marginal_log.append(self.nodes_out(message))
            delta = self.marginal_delta(marginal_log)
            message_delta.append(delta)
            if delta < 5E-5:
                break

        marginal = torch.stack(marginal_log, dim=0)
        return marginal, message_delta

    def nodes_out(self, message):
        message = self.node_lin(message)
        marginal = F.log_softmax(message, dim=1)
        return marginal
    
    def marginal_delta(self, marginal_log):
        if len(marginal_log)<=1:
            return 1000
        else:
            marginal_new = torch.exp(marginal_log[-1])
            marginal_old = torch.exp(marginal_log[-2])
            return torch.max(torch.abs(marginal_new-marginal_old)).item()


    
