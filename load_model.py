import torch
import torch.nn as nn
from constants import *
import torch.optim as optim
from edge_generator import EdgeGenerator
from matrix_generator import MatrixGenerator
from combined_generator import CombinedGenerator

num_rows = 16

# output_size_matrix = num_rows * num_rows 
#GENETATOR
input_size_gen = 100  # Size of the random noise vector for the generator
hidden_size_gen = 200
output_size_edge_gen = num_rows * num_rows   # Output size for the matrix used to generate edge in edge generator
#output_size_mat_gen = num_rows * 3   # Output size for the matrix generator
output_size_mat_gen = num_rows * 3

#DISCRIM
input_size_discrim = 16*19
output_size_discrim = 1
hidden_size_discrim = 100

class CombinedGenerator(nn.Module):
    def __init__(self, edge_generator, matrix_generator):
        super(CombinedGenerator, self).__init__()
        self.edge_generator = edge_generator
        self.matrix_generator = matrix_generator

    def forward(self, rand_noise):
        # Generate adj mat from input
        adj = self.edge_generator(rand_noise)
        
        # Generate full image matrix based on edges
        matrices = self.matrix_generator(rand_noise)
        # print(matrices)
        # print(adj)
        comb = torch.cat((matrices, adj), dim=1)
        return comb

class EdgeGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EdgeGenerator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )

    def forward(self, noise):
        matrix_for_edge = self.generator(noise)
        # Apply a binary step function (threshold at 0.5)
        matrix_for_edge = torch.where(matrix_for_edge >= 0, torch.tensor(1.0), torch.tensor(0.0))
        # last_col = torch.randint(low=1, high=55, size=(num_rows,))
        matrix_for_edge = torch.reshape(matrix_for_edge, (num_rows,num_rows))
        # last_col = last_col[:, None]
        # generated_matrix = torch.cat([generated_matrix, last_col], axis=-1)
        
        matrix_for_edge = matrix_for_edge.fill_diagonal_(0)
        # print(matrix_for_edge)
        # print("Edge list: ")
        # generated_edge_index = torch.nonzero(matrix_for_edge, as_tuple=False).t()
        return matrix_for_edge
        
class MatrixGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MatrixGenerator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
    def forward(self, noise):
        gen_matrix = self.generator(noise)

        gen_matrix = torch.reshape(gen_matrix, (num_rows,3))

        return gen_matrix
        
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))
        return out

def load_model():
    edge_generator = EdgeGenerator(INPUT_SIZE_GEN, HIDDEN_SIZE_GEN, OUTPUT_SIZE_EDGE_GEN)
    matrix_generator = MatrixGenerator(INPUT_SIZE_GEN, HIDDEN_SIZE_GEN, OUTPUT_SIZE_MAT_GEN)

    model = CombinedGenerator(edge_generator, matrix_generator)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    def get_fake_data(batch_size, combined):
        fake_graphs=[]
        for i in range(batch_size):
            rand_noise = torch.randn(1, INPUT_SIZE_GEN)
            fake_graphs.append(combined(rand_noise))
        return fake_graphs 
    fake_data = get_fake_data(64,model)

    def deconstructor(matrix):
        mat1 = matrix[:, :3]  
        mat2 = matrix[:, 3:]  
        return mat1, mat2

    def adj_matrix_to_dict(adj_matrix):
        adj_dict = {}
        for i, row in enumerate(adj_matrix):
            adj_dict[i] = []
            for j, edge in enumerate(row):
                if edge != 0:
                    adj_dict[i].append(j)
        return adj_dict

    dict_list = []
    for data in fake_data:
        dict_list.append(adj_matrix_to_dict(deconstructor(data)[1]))
    return dict_list