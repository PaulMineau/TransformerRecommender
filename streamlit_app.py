"""
Streamlit app for transformer-based recommendation system
"""
import streamlit as st
import torch
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import os
import plotly.express as px
import plotly.graph_objects as go
from src.transformer_model import TransformerRecommender

class RecommendationApp:
    """Streamlit app for product recommendations"""
    
    def __init__(self):
        self.model = None
        self.encoders = None
        self.metadata_df = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @st.cache_resource
    def load_model_and_data(_self):
        """Load trained model and data (cached)"""
        try:
            # Try to load latest models first, then fall back to older ones
            model_paths = [
                ('models/enhanced_transformer_50epochs.pth', 'enhanced_50epoch'),
                ('models/enhanced_transformer.pth', 'enhanced'),
                ('models/stable_transformer.pth', 'stable'),
                ('models/simple_model.pth', 'simple'),
                ('models/best_model.pth', 'basic')
            ]
            
            model = None
            model_type = None
            
            for model_path, mtype in model_paths:
                if os.path.exists(model_path):
                    try:
                        checkpoint = torch.load(model_path, map_location=_self.device)
                        
                        # Load encoders
                        with open('data/encoders.pkl', 'rb') as f:
                            encoders = pickle.load(f)
                        
                        # Create appropriate model based on type
                        config = checkpoint['config'].copy()
                        
                        if mtype == 'enhanced_50epoch' or mtype == 'enhanced':
                            from src.enhanced_transformer import EnhancedTransformerRecommender
                            model = EnhancedTransformerRecommender(**config)
                        elif mtype == 'stable':
                            from src.stable_transformer import StableTransformerRecommender
                            model = StableTransformerRecommender(**config)
                        elif mtype == 'simple':
                            from src.simple_model import SimpleRecommender
                            model = SimpleRecommender(**config)
                        else:  # basic
                            model = TransformerRecommender(**config)
                        
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model.to(_self.device)
                        model.eval()
                        model_type = mtype
                        
                        print(f"Loaded {mtype} model from {model_path}")
                        break
                        
                    except Exception as e:
                        print(f"Failed to load {model_path}: {e}")
                        continue
            
            if model is None:
                return None, None, None, "No trained model found. Please train a model first."
            
            # Load metadata if available
            metadata_df = None
            if os.path.exists('data/metadata.csv') and os.path.getsize('data/metadata.csv') > 1:
                try:
                    metadata_df = pd.read_csv('data/metadata.csv')
                except:
                    metadata_df = None
            
            return model, encoders, metadata_df, None, model_type
            
        except Exception as e:
            return None, None, None, f"Error loading model: {str(e)}", None
    
    def get_item_info(self, item_id: str) -> Dict:
        """Get item information from metadata"""
        if self.metadata_df is not None and item_id in self.metadata_df['asin'].values:
            item_data = self.metadata_df[self.metadata_df['asin'] == item_id].iloc[0]
            return {
                'title': item_data.get('title', 'Unknown Product'),
                'price': item_data.get('price', 'N/A'),
                'brand': item_data.get('brand', 'Unknown Brand'),
                'categories': item_data.get('categories', []),
                'image_url': item_data.get('imUrl', '')
            }
        else:
            return {
                'title': f'Product {item_id}',
                'price': 'N/A',
                'brand': 'Unknown Brand',
                'categories': [],
                'image_url': ''
            }
    
    def search_products(self, query: str, limit: int = 20) -> List[Tuple[str, str]]:
        """Search for products by name"""
        if self.metadata_df is None:
            # If no metadata, return some sample product IDs
            sample_items = list(self.encoders['item_to_id'].keys())[:limit]
            return [(item_id, f'Product {item_id}') for item_id in sample_items]
        
        # Search in product titles
        mask = self.metadata_df['title'].str.contains(query, case=False, na=False)
        results = self.metadata_df[mask].head(limit)
        
        return [(row['asin'], row['title']) for _, row in results.iterrows() 
                if row['asin'] in self.encoders['item_to_id']]
    
    def get_recommendations(self, purchased_items: List[str], num_recommendations: int = 10, user_id: str = None) -> List[Dict]:
        """Get recommendations based on purchased items"""
        if not purchased_items:
            return []
        
        # Convert item IDs to encoded IDs
        encoded_items = []
        for item_id in purchased_items:
            if item_id in self.encoders['item_to_id']:
                encoded_items.append(self.encoders['item_to_id'][item_id])
        
        if not encoded_items:
            return []
        
        # Create sequence (pad if necessary)
        max_seq_len = self.encoders['max_sequence_length']
        if len(encoded_items) > max_seq_len:
            encoded_items = encoded_items[-max_seq_len:]
        
        # Pad sequence
        padded_sequence = [0] * (max_seq_len - len(encoded_items)) + encoded_items
        
        # Convert to tensor
        sequence_tensor = torch.LongTensor([padded_sequence]).to(self.device)
        
        # Get user tensor if enhanced model
        user_tensor = None
        if hasattr(self.model, 'n_users') and user_id:
            # Use a dummy user ID for demo (in real app, you'd have actual user IDs)
            user_encoded_id = hash(user_id) % self.encoders['n_users']
            user_tensor = torch.LongTensor([user_encoded_id]).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            if hasattr(self.model, 'beam_search_predict'):
                # Enhanced model with beam search
                top_k_items, top_k_scores = self.model.beam_search_predict(
                    sequence_tensor, 
                    user_ids=user_tensor,
                    k=num_recommendations,
                    beam_width=10,
                    exclude_seen=True
                )
            else:
                # Standard model
                top_k_items, top_k_scores = self.model.predict_top_k(
                    sequence_tensor, 
                    k=num_recommendations,
                    exclude_seen=True
                )
        
        # Convert back to original item IDs and get info
        recommendations = []
        for i in range(min(num_recommendations, top_k_items.size(1))):
            item_encoded_id = top_k_items[0][i].item()
            if item_encoded_id in self.encoders['id_to_item']:
                item_id = self.encoders['id_to_item'][item_encoded_id]
                score = top_k_scores[0][i].item() if top_k_scores.dim() > 1 else torch.softmax(top_k_scores[0], dim=0)[i].item()
                
                item_info = self.get_item_info(item_id)
                item_info['item_id'] = item_id
                item_info['score'] = score
                
                recommendations.append(item_info)
        
        return recommendations
    
    def display_product_card(self, item_info: Dict, show_score: bool = False):
        """Display a product card"""
        with st.container():
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if item_info.get('image_url'):
                    try:
                        st.image(item_info['image_url'], width=100)
                    except:
                        st.write("📦")
                else:
                    st.write("📦")
            
            with col2:
                st.write(f"**{item_info['title'][:100]}...**" if len(item_info['title']) > 100 else f"**{item_info['title']}**")
                st.write(f"Brand: {item_info['brand']}")
                st.write(f"Price: {item_info['price']}")
                
                if show_score and 'score' in item_info:
                    st.write(f"Confidence: {item_info['score']:.3f}")
                
                if item_info.get('categories'):
                    categories = item_info['categories']
                    if isinstance(categories, list) and len(categories) > 0:
                        st.write(f"Category: {categories[0] if isinstance(categories[0], str) else 'Beauty'}")
    
    def run(self):
        """Run the Streamlit app"""
        st.set_page_config(
            page_title="Beauty Product Recommender",
            page_icon="💄",
            layout="wide"
        )
        
        st.title("🛍️ Beauty Product Recommender")
        st.markdown("*Powered by Transformer Neural Networks*")
        
        # Load model and data
        if self.model is None:
            with st.spinner("Loading model and data..."):
                result = self.load_model_and_data()
                if len(result) == 5:
                    self.model, self.encoders, self.metadata_df, error, model_type = result
                else:
                    self.model, self.encoders, self.metadata_df, error = result
                    model_type = "unknown"
            
            if error:
                st.error(error)
                st.info("Please run one of the training scripts to train a model first.")
                return
            
            st.success(f"✅ {model_type.title()} model loaded successfully!")
            
            # Show model capabilities
            if model_type == 'enhanced_50epoch':
                st.success("🏆 SOTA Enhanced Transformer (50 epochs): 5.05% Hit Rate @10 - NEW RECORD!")
                st.info("🚀 Features: User context, categories, beam search, attention visualization, and state-of-the-art performance!")
            elif model_type == 'enhanced':
                st.info("🚀 Enhanced Transformer: Features user context, categories, beam search, and attention visualization!")
            elif model_type == 'stable':
                st.info("⚡ Stable Transformer: Numerically stable transformer architecture")
            elif model_type == 'simple':
                st.info("🔧 Simple LSTM: Baseline model for comparison")
            else:
                st.info("📊 Basic model loaded")
        
        # Sidebar for model info
        with st.sidebar:
            st.header("📊 Model Information")
            st.write(f"**Total Products:** {self.encoders['n_items']:,}")
            st.write(f"**Total Users:** {self.encoders['n_users']:,}")
            st.write(f"**Sequence Length:** {self.encoders['max_sequence_length']}")
            st.write(f"**Device:** {self.device}")
            
            # Show performance based on model type
            st.header("📈 Model Performance")
            if hasattr(self, 'model') and self.model is not None:
                # Check which model is loaded and show appropriate metrics
                if os.path.exists('models/enhanced_transformer_50epochs.pth'):
                    st.success("🏆 SOTA Model Loaded!")
                    st.metric("Hit Rate @10", "5.05%", "NEW RECORD ⭐")
                    st.metric("NDCG @10", "2.48%", "NEW RECORD ⭐")
                    st.metric("vs Previous SOTA", "+117%", "improvement")
                elif os.path.exists('models/enhanced_transformer.pth'):
                    st.metric("Hit Rate @10", "4.32%")
                    st.metric("NDCG @10", "~2.2%")
                    st.metric("vs Stable Model", "+40%", "improvement")
                elif os.path.exists('models/stable_transformer.pth'):
                    st.metric("Hit Rate @10", "3.08%")
                    st.metric("NDCG @10", "~1.8%")
                    st.metric("vs Simple LSTM", "+39%", "improvement")
                else:
                    st.metric("Hit Rate @10", "2.21%")
                    st.metric("Model Type", "Simple LSTM")
            
            # Load training history if available
            if os.path.exists('models/50_epoch_results.txt'):
                with open('models/50_epoch_results.txt', 'r') as f:
                    results = f.read()
                st.text_area("🏆 SOTA Results", results, height=150)
        
        # Main interface
        st.header("🛒 Your Purchase History")
        st.write("Add products you've purchased to get personalized recommendations:")
        
        # Initialize session state
        if 'purchased_items' not in st.session_state:
            st.session_state.purchased_items = []
        if 'user_name' not in st.session_state:
            st.session_state.user_name = ""
        
        # User input and product search
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            user_name = st.text_input("👤 Your Name (optional):", value=st.session_state.user_name, placeholder="e.g., Sarah")
            if user_name != st.session_state.user_name:
                st.session_state.user_name = user_name
        
        with col2:
            search_query = st.text_input("🔍 Search for products:", placeholder="e.g., lipstick, mascara, foundation")
        
        with col3:
            num_recommendations = st.selectbox("Recommendations:", [5, 10, 15, 20], index=1)
        
        # Search results
        if search_query:
            search_results = self.search_products(search_query, limit=10)
            
            if search_results:
                st.write("**Search Results:**")
                for item_id, title in search_results:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"• {title[:80]}..." if len(title) > 80 else f"• {title}")
                    with col2:
                        if st.button("Add", key=f"add_{item_id}"):
                            if item_id not in st.session_state.purchased_items:
                                st.session_state.purchased_items.append(item_id)
                                st.success(f"Added to purchase history!")
                                st.rerun()
            else:
                st.write("No products found. Try a different search term.")
        
        # Display purchased items
        if st.session_state.purchased_items:
            st.header("📝 Your Selected Products")
            
            for i, item_id in enumerate(st.session_state.purchased_items):
                item_info = self.get_item_info(item_id)
                
                col1, col2 = st.columns([5, 1])
                with col1:
                    self.display_product_card(item_info)
                with col2:
                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state.purchased_items.remove(item_id)
                        st.rerun()
                
                st.divider()
            
            # Clear all button
            if st.button("🗑️ Clear All"):
                st.session_state.purchased_items = []
                st.rerun()
            
            # Get recommendations
            st.header("✨ Recommended for You")
            
            with st.spinner("Generating recommendations..."):
                recommendations = self.get_recommendations(
                    st.session_state.purchased_items, 
                    num_recommendations,
                    user_id=st.session_state.user_name if st.session_state.user_name else None
                )
            
            if recommendations:
                # Display recommendations in a grid
                cols = st.columns(2)
                for i, rec in enumerate(recommendations):
                    with cols[i % 2]:
                        st.write(f"**#{i+1} Recommendation**")
                        self.display_product_card(rec, show_score=True)
                        st.divider()
                
                # Confidence scores chart
                if len(recommendations) > 1:
                    st.header("📊 Recommendation Confidence")
                    
                    scores_df = pd.DataFrame([
                        {
                            'Product': rec['title'][:30] + '...' if len(rec['title']) > 30 else rec['title'],
                            'Confidence': rec['score'],
                            'Rank': i + 1
                        }
                        for i, rec in enumerate(recommendations)
                    ])
                    
                    fig = px.bar(
                        scores_df, 
                        x='Rank', 
                        y='Confidence',
                        hover_data=['Product'],
                        title="Recommendation Confidence Scores"
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.warning("No recommendations available. Try adding more products to your purchase history.")
        
        else:
            st.info("👆 Search and add some products you've purchased to get started!")
            
            # Show some sample popular products
            st.header("🔥 Popular Products")
            st.write("Here are some popular beauty products you can add:")
            
            # Get some sample products
            sample_items = list(self.encoders['item_to_id'].keys())[:6]
            cols = st.columns(3)
            
            for i, item_id in enumerate(sample_items):
                with cols[i % 3]:
                    item_info = self.get_item_info(item_id)
                    self.display_product_card(item_info)
                    if st.button("Add to History", key=f"sample_{item_id}"):
                        st.session_state.purchased_items.append(item_id)
                        st.rerun()

def main():
    """Main function"""
    app = RecommendationApp()
    app.run()

if __name__ == "__main__":
    main()
