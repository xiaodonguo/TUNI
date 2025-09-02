import torch
import torch.nn as nn

class contrastive_loss(nn.Module):
    def __init__(self):
        super(contrastive_loss, self).__init__()

    def mag(self, x):
        x = x ** 2
        s = x.sum()
        s = s ** (1 / 2)
        return s

    def cosine_similarity(self, x, y):
        S = (x * y).sum()
        S = S / (self.mag(x) * self.mag(y))
        return S

    def forward(self, pos, neg, t=1):
        N, D = torch.zeros([1]).cuda(), torch.zeros([1]).cuda()  # .to(device)
        p = len(pos)
        for i in range(1, p):
            cos = self.cosine_similarity(pos[i], pos[0]).cuda()
            N += torch.exp(cos / t)
        # print(self.N)
        n = len(neg)
        for i in range(n):
            cos = self.cosine_similarity(pos[0], neg[i]).cuda()
            D += torch.exp(cos / t)
        # print(self.D)
        loss = - torch.log(N / (N + D))
        # N, D = convert(N), convert(D)
        return loss

class CL_Loss(nn.Module):
    def __init__(self):
        super(CL_Loss, self).__init__()

    def mag(self, x):
        x = x ** 2
        s = x.sum()
        s = s ** (1 / 2)
        return s

    def cosine_similarity(self, x, y):
        S = (x * y).sum()
        S = S / (self.mag(x) * self.mag(y))
        return S

    def forward(self, pre, pos, neg, t=1):
        N, D = torch.zeros([1]).cuda(), torch.zeros([1]).cuda()  # .to(device)

        cos = self.cosine_similarity(pre, pos).cuda()
        N += torch.exp(cos / t)
        # print(self.N)

        cos = self.cosine_similarity(pre, neg).cuda()
        D += torch.exp(cos / t)
        # print(self.D)
        loss = - torch.log(N / (N + D))
        # N, D = convert(N), convert(D)
        return loss

if __name__ == "__main__":
    # a = torch.randn(4, 3, 5, 5)
    # print(a)
    # b = a[1]
    # c = a[0]
    # print(b, b.shape, "\n", c, c.shape)

    a = torch.tensor([1, 2, 3, 4])
    embed_dim = 5
    embedding = nn.Embedding(a.shape[0], embed_dim)
    embedded = embedding(a)
    print(a)
    print(embedded, embedded.shape)
