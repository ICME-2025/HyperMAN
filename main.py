def main():
    data_generator = DataGenerator()
    check_ins_df = data_generator.generate_sample_data()
    
    data_processor = DataProcessor(check_ins_df)
    train_data, test_data = data_processor.split_data()
    
    hypergraph = HeterogeneousHypergraph(check_ins_df)
    
    model = HypergraphNN(
        num_users=len(hypergraph.users),
        num_pois=len(hypergraph.pois),
        num_categories=len(hypergraph.categories),
        num_time_slots=24,
        hidden_dim=64
    )
    
    meta_learner = DiversityAwareMetaLearning(model)
    
    num_epochs = 50
    for epoch in range(num_epochs):
        loss = meta_learner.train_step(hypergraph, train_data, test_data)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

if __name__ == "__main__":
    main()