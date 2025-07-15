import torch
import torch.nn as nn
from einops import einsum
from torch.utils import data as D
import transformer

class AttnVector(nn.Module):
    def __init__(self, size):
        super(AttnVector, self).__init__()
        # Define the unconstrained parameter
        self.unconstrained_param = nn.Parameter(torch.zeros(size))

    def forward(self):
        # Apply the sigmoid function to constrain the parameter between 0 and 1
        return torch.sigmoid(self.unconstrained_param)


class TCRPNet(torch.nn.Module):
    '''
    Input: 
    TCRb and epitope position wise embeddings + onehot -> (B, L, D + 20)
    TCRb and epitope single embeddings -> (B, D)

    Output:
    Predicted TCRb peptide interaction score
    '''
    def __init__(self, use_attn=False):
        super(TCRPNet, self).__init__()
        self.use_attn = use_attn
        if self.use_attn:
            self.attn_cdr = AttnVector(15)
            self.attn_peptide = AttnVector(9)
        
        self.conv_kernel = (5,5)
        self.pool_kernel = (5,5)

        d_model = 1280 + 20 # This should match the ESM embedding size
        num_heads = 10
        num_layers = 6
        d_ff = 2048
        dropout = 0.1
        max_len = 100

        self.encoder_tcrb = transformer.TransformerEncoder(d_model, num_heads, num_layers, d_ff, dropout, max_len)
        self.encoder_epitope = transformer.TransformerEncoder(d_model, num_heads, num_layers, d_ff, dropout, max_len)

        self.H1 = 32
        self.H2 = 64
        self.H3 = 128
        self.input_channels = 2

        self.conv_net = nn.Sequential(
            nn.Conv2d(self.input_channels, self.H1, kernel_size = self.conv_kernel),
            nn.MaxPool2d(kernel_size = self.pool_kernel),
            nn.ReLU(),
            nn.InstanceNorm2d(self.H1),

            nn.Conv2d(self.H1, self.H2, kernel_size = self.conv_kernel),
            nn.MaxPool2d(kernel_size = self.pool_kernel),
            nn.ReLU(),
            nn.InstanceNorm2d(self.H2),

            nn.Conv2d(self.H2, self.H3, kernel_size = self.conv_kernel),
            nn.MaxPool2d(kernel_size = self.pool_kernel),
            nn.ReLU(),
            nn.InstanceNorm2d(self.H3)
            )
        self.classifier = nn.Sequential(
            nn.Linear(9 * 9 * self.H3, self.H3),
            nn.ReLU(inplace = True),
            nn.Linear(self.H3,1),
            # nn.Sigmoid()
        )

    def forward(self,TCRb_pw, epitope_pw, TCRb_single, epitope_single, TCRb_mask=None, epitope_mask=None):
        if self.use_attn:
            TCRb_single = torch.clone(TCRb_pw[:,:,:-20])
            _, TCRb_el, _ = TCRb_single.shape
            TCRb_single = TCRb_single * self.attn_cdr().unsqueeze(0).unsqueeze(-1)
            TCRb_single = TCRb_single.sum(1)/TCRb_el
            epitope_single = torch.clone(epitope_pw[:,:,:-20])
            _, epitope_el, _ = epitope_single.shape
            epitope_single = epitope_single * self.attn_peptide().unsqueeze(0).unsqueeze(-1)
            epitope_single = epitope_single.sum(1)/epitope_el
            
        # Encode pw embedding
        # print('TCR')
        TCRb_encoded = self.encoder_tcrb(TCRb_pw, mask=TCRb_mask)
        # print('peptide')
        epitope_encoded = self.encoder_epitope(epitope_pw, mask=epitope_mask)
        
        # compute pair rep from both encoded pw embedding and single embedding
        outers_encoded = []
        outers_single = []
        for i in range(TCRb_single.size(0)):
            outer = einsum(TCRb_single[i,:].unsqueeze(0), epitope_single[i,:].unsqueeze(0), 'a b, c d -> a b d').unsqueeze(1)
            outers_single.append(outer)
        for i in range(TCRb_encoded.size(0)):
            outer = einsum(TCRb_encoded[i,:].unsqueeze(0), epitope_encoded[i,:].unsqueeze(0), 'a b, c d -> a b d').unsqueeze(1)
            outers_encoded.append(outer)
        single_pair = torch.cat(outers_single,dim = 0)
        encoded_pair = torch.cat(outers_encoded, dim = 0)
        
        # 2D output network for prediction
        out = torch.cat([single_pair, encoded_pair], dim = 1)
        out = self.conv_net(out)
        out = out.view(out.size(0), out.size(-1) * out.size(-2) * self.H3)
        out = self.classifier(out)
        return out

if __name__ == '__main__':
    import data_loader
    model = TCRPNet()

    TCRb_pw = torch.randn((8, 20, 1300))
    epitope_pw = torch.randn((8, 9, 1300))

    TCRb_single = torch.randn((8, 1280))
    epitope_single = torch.randn((8, 1280))

    out = model(TCRb_pw = TCRb_pw, epitope_pw = epitope_pw, TCRb_single = TCRb_single, epitope_single = epitope_single)

    '''
    Dset = data_loader.data_loader(partition = 'train')
    loader = D.DataLoader(Dset, batch_size = 8, num_workers=20, shuffle=True)
    for TCRb, epitope, label, _, _ in loader:
        out = model(TCRb, epitope)
        print(out.size())
        break
        #print(infeat.size())
        #break
        #out = model()
    '''
