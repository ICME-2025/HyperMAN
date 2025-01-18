class HypergraphNN(torch.nn.Module):
    def __init__(self, num_users, num_pois, num_categories, num_time_slots, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.user_embedding = torch.nn.Embedding(num_users, hidden_dim)
        self.poi_embedding = torch.nn.Embedding(num_pois, hidden_dim)
        self.category_embedding = torch.nn.Embedding(num_categories, hidden_dim)
        self.time_embedding = torch.nn.Embedding(num_time_slots, hidden_dim)
        
        self.W_temporal = torch.nn.Linear(3 * hidden_dim, hidden_dim)
        self.W_spatial = torch.nn.Linear(3 * hidden_dim, hidden_dim)
        self.W_preference = torch.nn.Linear(3 * hidden_dim, hidden_dim)
        
        self.attention = torch.nn.Linear(3 * hidden_dim, 1)
        self.output_layer = torch.nn.Linear(3 * hidden_dim, num_pois)
        
        self.degree_matrices = {}
    
    def compute_degree_matrices(self, hypergraph):
        edge_types = ['temporal', 'spatial', 'preference']
        for edge_type in edge_types:
            edges = getattr(hypergraph, f"{edge_type}_edges")
            vertex_degrees = torch.zeros(self.num_nodes)
            edge_degrees = torch.zeros(len(edges))
            
            for i, edge in enumerate(edges):
                if edge_type == 'temporal':
                    nodes = edge[:3]
                elif edge_type == 'spatial':
                    nodes = edge
                else:
                    nodes = [edge[0]] + edge[1] + [edge[2]]
                
                for node in nodes:
                    vertex_degrees[node] += 1
                edge_degrees[i] = len(nodes)
            
            D_v = torch.diag(1.0 / torch.sqrt(vertex_degrees + 1e-10))
            D_e = torch.diag(1.0 / (edge_degrees + 1e-10))
            self.degree_matrices[edge_type] = (D_v, D_e)
    
    def forward(self, hypergraph, trajectory):
        self.compute_degree_matrices(hypergraph)
        
        node_embeddings = self.compute_node_embeddings()
        message_embeddings = self.compute_messages(hypergraph, node_embeddings)
        trajectory_repr = self.compute_trajectory_representation(trajectory, message_embeddings)
        
        return self.compute_prediction(trajectory_repr)
    
    def compute_node_embeddings(self):
        return {
            'user': self.user_embedding.weight,
            'poi': self.poi_embedding.weight,
            'category': self.category_embedding.weight,
            'time': self.time_embedding.weight
        }
    
    def compute_messages(self, hypergraph, node_embeddings):
        temporal_msg = self.temporal_message_passing(hypergraph, node_embeddings)
        spatial_msg = self.spatial_message_passing(hypergraph, node_embeddings)
        preference_msg = self.preference_message_passing(hypergraph, node_embeddings)
        
        return temporal_msg + spatial_msg + preference_msg
    
    def compute_trajectory_representation(self, trajectory, node_embeddings):
        trajectory_features = []
        for p, c, t in trajectory:
            combined = torch.cat([
                node_embeddings['poi'][p],
                node_embeddings['category'][c],
                node_embeddings['time'][t]
            ])
            trajectory_features.append(combined)
        
        trajectory_features = torch.stack(trajectory_features)
        attention_weights = torch.softmax(self.attention(trajectory_features), dim=0)
        
        return (attention_weights * trajectory_features).sum(dim=0)
    
    def compute_prediction(self, trajectory_repr):
        logits = self.output_layer(trajectory_repr)
        return torch.softmax(logits, dim=-1)