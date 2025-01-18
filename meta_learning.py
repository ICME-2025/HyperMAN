class DiversityAwareMetaLearning:
    def __init__(self, model, base_lr=0.01, beta=1.0):
        self.model = model
        self.base_lr = base_lr
        self.beta = beta
        self.optimizer = torch.optim.Adam(model.parameters())
        
    def compute_diversity(self, user_checkins):
        category_dist = user_checkins['category'].value_counts(normalize=True)
        return -torch.sum(category_dist * torch.log(category_dist + 1e-10))
    
    def compute_adaptive_learning_rate(self, entropy):
        return self.base_lr * torch.sigmoid(self.beta * entropy)
    
    def train_step(self, hypergraph, support_set, query_set):
        meta_loss = 0
        
        for user in support_set:
            entropy = self.compute_diversity(support_set[user]['data'])
            alpha = self.compute_adaptive_learning_rate(entropy)
            
            self.inner_update(support_set[user], hypergraph, alpha)
            meta_loss += self.outer_update(query_set[user], hypergraph)
        
        return meta_loss / len(support_set)
    
    def inner_update(self, user_data, hypergraph, learning_rate):
        self.optimizer.zero_grad()
        loss = self.compute_loss(user_data['trajectories'], hypergraph)
        loss.backward()
        
        for param in self.model.parameters():
            param.data -= learning_rate * param.grad
    
    def outer_update(self, user_data, hypergraph):
        self.optimizer.zero_grad()
        loss = self.compute_loss(user_data['trajectories'], hypergraph)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def compute_loss(self, trajectories, hypergraph):
        total_loss = 0
        for trajectory, next_poi in trajectories:
            pred = self.model(hypergraph, trajectory)
            total_loss += torch.nn.functional.cross_entropy(
                pred.unsqueeze(0),
                torch.tensor([next_poi])
            )
        return total_loss / len(trajectories)