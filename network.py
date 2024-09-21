import torch.nn as nn
import torch.nn.functional as F 
import torch

class single_head_network(nn.Module): 
    def __init__(self, input_dim, out_dim, hidden_dim, n_layers): 
        super( ).__init__()
        self.hidden_layes = hidden_dim
        self.nblk  =  n_layers
        self.start = nn.Linear(input_dim, hidden_dim)
        self.layer =  nn.Linear( hidden_dim, hidden_dim)
        self.out =  nn.Linear( hidden_dim, out_dim)
        self.mod_list = nn.ModuleList()
        for _ in range(self.nblk): 
            self.mod_list.append(self.layer)
        
    def forward(self,x): 
        x = F.relu(self.start(x))
        for layers in self.mod_list: 
            x = F.relu(layers(x))

        out = F.relu(self.out(x))
        return out # return the logit for later use
    
class two_head_output (nn.Module): 
    def __init__(self, input_dim, out_dim, hidden_dim, n_layes): 
        super( ).__init__()
        self.H_net =  single_head_network(input_dim, out_dim, hidden_dim, n_layes )
        self.E_net =  single_head_network(input_dim, out_dim, hidden_dim, n_layes )
        self.critic =  single_head_network(input_dim*2, 1, hidden_dim, n_layes )
    def forward( self, x): 
        if len( x.shape) > 1: 
            batch, nfeature =  x.shape
            x_1, x_2 = x[:, : int(nfeature/2)], x[:, int( nfeature/2):]
        else: 
            x_1, x_2 = x[: int(len(x)/2)], x[int(len(x) /2):]
        
        output_x1 = self.H_net(x_1)
        output_x2 = self.E_net(x_2)
        critic_output = self.critic(x)
        actor_output = F.softmax( torch.cat([output_x1, output_x2], dim=-1), dim = -1)

        return actor_output, critic_output

  

