import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),    # 输入通道数1，输出通道数20
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),          # 最大池化，核大小为2，步长为2
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),     # 全连接层：输入的二维张量大小：50*4*4， 输出的二维张量大小：500
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),         # 全连接层
            nn.Tanh(),                         # Tanh 激活
            nn.Linear(self.D, self.K)          # 全连接层
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)                                # 压缩维度,11个示例->11维
        H = self.feature_extractor_part1(x)             # 卷积池化，卷积池化
        H = H.view(-1, 50 * 4 * 4)                      # resize()
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxAK                    # attention 机制: 全连接+Tanh激活+全连接
        A = torch.transpose(A, 1, 0)  # KxN             # 交换tensor的两个维度
        A = F.softmax(A, dim=1)  # softmax over N       # 按行softmax
        M = torch.mm(A, H)  # KxL                       # 矩阵相乘：A:N*K, H:K*L
        Y_prob = self.classifier(M)                     # 分类： 全连接 + sigmoid
        Y_hat = torch.ge(Y_prob, 0.5).float()           # Y_prob 是否大于0.5

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        # error = 1. - Y_hat.eq(Y).cpu().float().mean().data[0]
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)   # 输入input张量每个元素的夹紧到区间 [min,max]，并返回结果到一个新张量。
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli     # 交叉熵伯努利

        return neg_log_likelihood, A

class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(  # 全连接 + tanh
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)                      # 卷积
        H = H.view(-1, 50 * 4 * 4)                               # resize()
        H = self.feature_extractor_part2(H)  # NxL               # 全连接

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN                     # 交换tensor的两个维度
        A = F.softmax(A, dim=1)  # softmax over N               # 按行sortmax

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
