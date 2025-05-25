import http.client
import json
import re
import urllib.parse

import nltk
import numpy as np
import requests
import spacy
import streamlit as st
from fuzzywuzzy import fuzz
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Set your RapidAPI key
RAPIDAPI_KEY = "RAPIDAPI_KEY"


def fuzzy_search_and_filter(query, products, threshold=50):
    filtered_products = []

    for product in products:
        fuzzy_score = fuzz.token_sort_ratio(query.lower(), product["name"].lower())

        if fuzzy_score >= threshold:
            product["fuzzy_score"] = fuzzy_score
            filtered_products.append(product)

    filtered_products.sort(key=lambda x: x["fuzzy_score"], reverse=True)

    return filtered_products


def search_ebay(query, rapidapi_key):
    try:
        query = urllib.parse.quote(query)
        conn = http.client.HTTPSConnection("real-time-ebay-data.p.rapidapi.com")
        query_url = urllib.parse.quote(f"https://www.ebay.com/sch/i.html?_nkw={query}")
        headers = {
            "x-rapidapi-host": "real-time-ebay-data.p.rapidapi.com",
            "x-rapidapi-key": rapidapi_key,
        }

        complete_url = f"https://www.ebay.com/sch/i.html?_nkw={query}"
        print(f"eBay Search URL: {complete_url}")

        conn.request("GET", f"/search_get.php?url={query_url}", headers=headers)

        res = conn.getresponse()
        print("----------eBay response:", res)
        data = res.read()

        if not data:
            return []

        try:
            response_data = json.loads(data.decode("utf-8"))
        except json.decoder.JSONDecodeError:
            return []

        if "products" in response_data.get("body", {}):
            products = []
            for item in response_data["body"]["products"]:
                name = item.get("title", "No Title")
                category = item.get("category", "No Category")
                brand = item.get("brand", "No Brand")
                url = item.get("url", "No URL")
                image = item.get("image", "No Image")

                price = item.get("price", {})
                price_from = (
                    price.get("from", "Price Not Available")
                    if isinstance(price, dict)
                    else price
                )

                products.append(
                    {
                        "name": name,
                        "category": category,
                        "brand": brand,
                        "url": url,
                        "image": image,
                        "price": price_from,
                    }
                )

            filtered_products = fuzzy_search_and_filter(query, products, threshold=50)
            return filtered_products
        else:
            return []
    except Exception as e:
        st.error(f"eBay API error: {str(e)}")
        return []


def search_walmart_products(product_name, threshold=50):
    url = "https://realtime-walmart-data.p.rapidapi.com/search"
    headers = {
        "x-rapidapi-host": "realtime-walmart-data.p.rapidapi.com",
        "x-rapidapi-key": "1049b79e85msh017342b0ecf6a28p12322ejsn2dd9e47fd727",
    }
    params = {"keyword": product_name}

    complete_url = f"{url}?{urllib.parse.urlencode(params)}"
    print(f"Walmart Search URL: {complete_url}")

    response = requests.get(url, headers=headers, params=params)
    print("----------Walmart response:", response)

    data = response.json()
    products = []
    if "results" in data:
        for item in data["results"]:
            price = item.get("price", "Price Not Available")
            products.append(
                {
                    "name": item["name"],
                    "category": item.get("category", "No Category"),
                    "brand": item.get("sellerName", "No Brand"),
                    "url": item["canonicalUrl"],
                    "image": item["image"],
                    "price": price,
                }
            )

    filtered_products = fuzzy_search_and_filter(product_name, products, threshold)
    return filtered_products


def search_amazon(query, rapidapi_key, threshold=50):
    try:
        url = "https://real-time-amazon-data.p.rapidapi.com/search"
        headers = {
            "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com",
            "x-rapidapi-key": rapidapi_key,
        }
        params = {
            "query": query,
            "page": 1,
            "sort_by": "RELEVANCE",
        }

        complete_url = f"{url}?{urllib.parse.urlencode(params)}"
        print(f"Amazon Search URL: {complete_url}")

        response = requests.get(url, headers=headers, params=params)
        print("----------Amazon response:", response)

        if response.status_code == 200:
            data = response.json()
            products = []
            if "data" in data and "products" in data["data"]:
                for item in data["data"]["products"]:
                    price = item.get("product_price", "Price Not Available")
                    if isinstance(price, dict):
                        price = price.get("value", "Price Not Available")

                    products.append(
                        {
                            "name": item["product_title"],
                            "category": item.get("category", "No Category"),
                            "brand": item.get("brand", "No Brand"),
                            "url": item["product_url"],
                            "image": item["product_photo"],
                            "price": price,
                        }
                    )

                filtered_products = fuzzy_search_and_filter(query, products, threshold)
                return filtered_products
            else:
                return []
        else:
            return []
    except Exception as e:
        st.error(f"Amazon API error: {str(e)}")
        return []


def get_image_features_resnet(image_url):
    try:
        img = Image.open(requests.get(image_url, stream=True).raw)
        img = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img), axis=0)
        processed_img = preprocess_input(img_array)

        model = ResNet50(weights="imagenet", include_top=False)
        features = model.predict(processed_img)
        return features.flatten()
    except Exception as e:
        print(f"Error extracting image features with ResNet: {e}")
        return None


def get_most_similar_product_from_platform(
    selected_product, platform_products, platform_name
):
    """Find the most similar product from a specific platform's products"""
    selected_image_features = get_image_features_resnet(selected_product["image"])

    if not platform_products or selected_image_features is None:
        return None, -1, platform_name

    highest_similarity = -1
    most_similar_product = None

    for product in platform_products:
        product_image_features = get_image_features_resnet(product["image"])

        if product_image_features is not None:
            similarity = cosine_similarity(
                [selected_image_features], [product_image_features]
            )[0][0]

            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_product = product

    return most_similar_product, highest_similarity, platform_name


st.title("Product Search Across Amazon, eBay, and Walmart")
st.write("Search for products and compare results from Amazon, eBay, and Walmart.")

search_query = st.text_input("Enter product to search:")

if st.button("Search"):
    with st.spinner("Searching across platforms..."):
        amazon_products = search_amazon(search_query, RAPIDAPI_KEY)
        ebay_products = search_ebay(search_query, RAPIDAPI_KEY)
        walmart_products = search_walmart_products(search_query)

        st.session_state.amazon_products = amazon_products
        st.session_state.ebay_products = ebay_products
        st.session_state.walmart_products = walmart_products

if (
    "amazon_products" in st.session_state
    and "ebay_products" in st.session_state
    and "walmart_products" in st.session_state
):
    st.header("Search Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Amazon Product")
        if st.session_state.amazon_products:
            product = st.session_state.amazon_products[0]
            st.image(product["image"], width=250)
            st.write(f"**{product['name']}**")
            st.write(f"**Price:** {product['price']}")
            st.write(f"**Brand:** {product.get('brand', 'Not specified')}")
            st.write(f"[View Product]({product['url']})")

            if st.button("Select Amazon Product", key="select_amazon"):
                st.session_state.selected_product = product
                st.session_state.selected_platform = "Amazon"
        else:
            st.warning("No Amazon products found")

    with col2:
        st.subheader("eBay Product")
        if st.session_state.ebay_products:
            product = st.session_state.ebay_products[0]
            if product["image"] != "No Image":
                st.image(product["image"], width=250)
            st.write(f"**{product['name']}**")
            st.write(f"**Price:** {product['price']}")
            st.write(f"**Brand:** {product.get('brand', 'Not specified')}")
            st.write(f"[View Product]({product['url']})")

            if st.button("Select eBay Product", key="select_ebay"):
                st.session_state.selected_product = product
                st.session_state.selected_platform = "eBay"
        else:
            st.warning("No eBay products found")

    with col3:
        st.subheader("Walmart Product")
        if st.session_state.walmart_products:
            product = st.session_state.walmart_products[0]
            st.image(product["image"], width=250)
            st.write(f"**{product['name']}**")
            st.write(f"**Price:** {product['price']}")
            st.write(f"**Brand:** {product.get('brand', 'Not specified')}")
            st.write(f"[View Product]({product['url']})")

            if st.button("Select Walmart Product", key="select_walmart"):
                st.session_state.selected_product = product
                st.session_state.selected_platform = "Walmart"
        else:
            st.warning("No Walmart products found")

if "selected_product" in st.session_state:
    st.header("Selected Product Details")
    selected = st.session_state.selected_product

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(selected["image"], width=250)

    with col2:
        st.subheader(selected["name"])
        st.write(f"**Platform:** {st.session_state.selected_platform}")
        st.write(f"**Price:** {selected['price']}")
        st.write(f"**Brand:** {selected.get('brand', 'Not specified')}")
        st.write(f"**Category:** {selected.get('category', 'Not specified')}")
        st.write(f"[View Original Product]({selected['url']})")

    st.subheader("Extracted Product Features")

    st.subheader(f"Find Similar Products on Other Platforms")

    if st.button("Find Most Similar Products"):
        with st.spinner(f"Searching for similar products..."):
            similar_products_results = []

            if st.session_state.selected_platform == "Amazon":
                similar_ebay_products = search_ebay(selected["name"], RAPIDAPI_KEY)
                similar_walmart_products = search_walmart_products(selected["name"])
                print("Products found on eBay:", similar_ebay_products)
                print("Products found on Walmart:", similar_walmart_products)
                st.write(
                    f"Number of products found on eBay: {len(similar_ebay_products)}"
                )
                st.write(
                    f"Number of products found on Walmart: {len(similar_walmart_products)}"
                )

                if similar_ebay_products:
                    ebay_similar, ebay_similarity, _ = (
                        get_most_similar_product_from_platform(
                            selected, similar_ebay_products, "eBay"
                        )
                    )
                    if ebay_similar:
                        similar_products_results.append(
                            (ebay_similar, ebay_similarity, "eBay")
                        )

                if similar_walmart_products:
                    walmart_similar, walmart_similarity, _ = (
                        get_most_similar_product_from_platform(
                            selected, similar_walmart_products, "Walmart"
                        )
                    )
                    if walmart_similar:
                        similar_products_results.append(
                            (walmart_similar, walmart_similarity, "Walmart")
                        )

            elif st.session_state.selected_platform == "eBay":
                similar_amazon_products = search_amazon(selected["name"], RAPIDAPI_KEY)
                similar_walmart_products = search_walmart_products(selected["name"])
                print("Products found on Amazon:", similar_amazon_products)
                print("Products found on Walmart:", similar_walmart_products)
                st.write(
                    f"Number of products found on Amazon: {len(similar_amazon_products)}"
                )
                st.write(
                    f"Number of products found on Walmart: {len(similar_walmart_products)}"
                )

                if similar_amazon_products:
                    amazon_similar, amazon_similarity, _ = (
                        get_most_similar_product_from_platform(
                            selected, similar_amazon_products, "Amazon"
                        )
                    )
                    if amazon_similar:
                        similar_products_results.append(
                            (amazon_similar, amazon_similarity, "Amazon")
                        )

                if similar_walmart_products:
                    walmart_similar, walmart_similarity, _ = (
                        get_most_similar_product_from_platform(
                            selected, similar_walmart_products, "Walmart"
                        )
                    )
                    if walmart_similar:
                        similar_products_results.append(
                            (walmart_similar, walmart_similarity, "Walmart")
                        )

            elif st.session_state.selected_platform == "Walmart":
                similar_amazon_products = search_amazon(selected["name"], RAPIDAPI_KEY)
                similar_ebay_products = search_ebay(selected["name"], RAPIDAPI_KEY)
                print("Products found on Amazon:", similar_amazon_products)
                print("Products found on eBay:", similar_ebay_products)
                st.write(
                    f"Number of products found on Amazon: {len(similar_amazon_products)}"
                )
                st.write(
                    f"Number of products found on eBay: {len(similar_ebay_products)}"
                )

                if similar_amazon_products:
                    amazon_similar, amazon_similarity, _ = (
                        get_most_similar_product_from_platform(
                            selected, similar_amazon_products, "Amazon"
                        )
                    )
                    if amazon_similar:
                        similar_products_results.append(
                            (amazon_similar, amazon_similarity, "Amazon")
                        )

                if similar_ebay_products:
                    ebay_similar, ebay_similarity, _ = (
                        get_most_similar_product_from_platform(
                            selected, similar_ebay_products, "eBay"
                        )
                    )
                    if ebay_similar:
                        similar_products_results.append(
                            (ebay_similar, ebay_similarity, "eBay")
                        )

            if similar_products_results:
                st.subheader("Most Similar Products by Platform")

                similar_products_results.sort(key=lambda x: x[1], reverse=True)

                for product, similarity, platform in similar_products_results:
                    st.write(f"### Most Similar Product from {platform}")

                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.image(product["image"], width=200)

                    with col2:
                        st.write(f"**Name:** {product['name']}")
                        st.write(f"**Price:** {product['price']}")
                        st.write(f"**Brand:** {product.get('brand', 'Not specified')}")
                        st.write(f"**Similarity Score:** {similarity:.4f}")
                        st.write(f"[View Product]({product['url']})")

                    st.write("---")
            else:
                st.warning("No similar products found on other platforms.")

