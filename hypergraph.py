class HeterogeneousHypergraph:
    def __init__(self, check_ins_df, distance_threshold=0.01):
        self.df = check_ins_df
        self.distance_threshold = distance_threshold
        self.construct_mappings()
        self.construct_hyperedges()
        
    def construct_mappings(self):
        self.users = sorted(self.df['user'].unique())
        self.pois = sorted(self.df['poi'].unique())
        self.categories = sorted(self.df['category'].unique())
        self.time_slots = list(range(24))
        
        self.user_to_idx = {u: i for i, u in enumerate(self.users)}
        self.poi_to_idx = {p: i for i, p in enumerate(self.pois)}
        self.category_to_idx = {c: i for i, c in enumerate(self.categories)}
    
    def construct_hyperedges(self):
        self.temporal_edges = self._construct_temporal_edges()
        self.spatial_edges = self._construct_spatial_edges()
        self.preference_edges = self._construct_preference_edges()
    
    def _construct_temporal_edges(self):
        edges = []
        for _, row in self.df.iterrows():
            edges.append((
                self.poi_to_idx[row['poi']],
                self.category_to_idx[row['category']],
                row['timestamp'].hour
            ))
        return edges
    
    def _construct_spatial_edges(self):
        edges = []
        for i, poi1 in enumerate(self.pois):
            for poi2 in self.pois[i+1:]:
                coord1 = self.df[self.df['poi'] == poi1]['coords'].iloc[0]
                coord2 = self.df[self.df['poi'] == poi2]['coords'].iloc[0]
                dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))
                
                if (dist <= self.distance_threshold and
                    self.df[self.df['poi'] == poi1]['category'].iloc[0] ==
                    self.df[self.df['poi'] == poi2]['category'].iloc[0]):
                    edges.append((
                        self.poi_to_idx[poi1],
                        self.poi_to_idx[poi2],
                        self.category_to_idx[self.df[self.df['poi'] == poi1]['category'].iloc[0]]
                    ))
        return edges
    
    def _construct_preference_edges(self):
        edges = []
        for user in self.users:
            user_data = self.df[self.df['user'] == user]
            for category in self.categories:
                category_pois = user_data[user_data['category'] == category]['poi'].unique()
                if len(category_pois) > 0:
                    edges.append((
                        self.user_to_idx[user],
                        [self.poi_to_idx[p] for p in category_pois],
                        self.category_to_idx[category]
                    ))
        return edges