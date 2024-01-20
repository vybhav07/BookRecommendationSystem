import streamlit as st
import pickle
import pandas as pd
from surprise import Reader, Dataset, KNNBasic

# Loading the final_ratings and books datasets
final_ratings = pd.read_csv("final_ratings.csv")
books = pd.read_csv("books.csv",encoding="latin-1")

# Loading the User-Item model from the pickle file
with open('user_based_model.pkl', 'rb') as file:
    user_based_model = pickle.load(file)

# Loading the Item-Item model from the pickle file
with open('item_based_model.pkl', 'rb') as file:
    item_based_model = pickle.load(file)

# Function to get top N recommendations for a user
def get_user_recommendations(user_id, top_n=5):
    items_to_recommend = [item for item in user_based_model.trainset.build_anti_testset() if item[0] == user_id]
    predicted_ratings = [user_based_model.predict(user_id, item[1]).est for item in items_to_recommend]
    item_ratings = list(zip([item[1] for item in items_to_recommend], predicted_ratings))
    item_ratings.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = item_ratings[:top_n]
    return top_recommendations

# Function to get top N recommendations for a book
def get_book_recommendations(book_title, top_n=5):
    item_id_to_recommend = user_based_model.trainset.to_inner_iid(book_title)
    similar_items = item_based_model.get_neighbors(item_id_to_recommend, k=top_n)

    recommendations = []
    
    for similar_item_id in similar_items:
        raw_iid = user_based_model.trainset.to_raw_iid(similar_item_id)
        book_info = books[books['Book-Title'] == raw_iid]
        
        if not book_info.empty:
            image_url = book_info.iloc[0]['Image-URL-L']
            recommendations.append((raw_iid, image_url))
        else:
            st.write(f"Book information not found for Book Title: {raw_iid}")

    return recommendations



# Streamlit app
def main():
    st.title("Book Recommendation System")

    # Page selection
    page = st.sidebar.selectbox("Select a page", ["User-Item Model", "Item-Item Model"])

    if page == "User-Item Model":
        st.header("User-Item Collaborative Filtering")

        # User input for User-Item recommendations
        user_id = st.selectbox("Select User ID", final_ratings['userid'].unique())
        top_n_user = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

        if st.button("Get Recommendations"):
            user_recommendations = get_user_recommendations(user_id, top_n=top_n_user)
            rows = (len(user_recommendations) - 1) // 5 + 1
            for i in range(rows):
                columns = st.columns(5)
                for col, (item_id, predicted_rating) in zip(columns, user_recommendations[i * 5: (i + 1) * 5]):
                    book_name = final_ratings[final_ratings['booktitle'] == item_id]['booktitle'].values[0]
                    book_info = books[books['Book-Title'] == book_name]
                    if not book_info.empty:
                        image_url = book_info.iloc[0]['Image-URL-L']
                        col.image(image_url, caption=f"Book Title: {book_name} | Predicted Rating: {predicted_rating}", use_column_width=True)
                    else:
                        st.write(f"Book information not found for Book Title: {book_name}")

    elif page == "Item-Item Model":
        st.header("Item-Item Collaborative Filtering")

        # User input for Item-Item recommendations
        book_title = st.selectbox("Select Book Title", final_ratings['booktitle'].unique())
        top_n_book = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

        if st.button("Get Recommendations"):
            book_recommendations = get_book_recommendations(book_title, top_n=top_n_book)
            rows = (len(book_recommendations) - 1) // 5 + 1
            for i in range(rows):
                columns = st.columns(5)
                for col, (book_id, image_url) in zip(columns, book_recommendations[i * 5: (i + 1) * 5]):
                    col.image(image_url, caption=f"Book Title: {book_id}", use_column_width=True)

if __name__ == "__main__":
    main()