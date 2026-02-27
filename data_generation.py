"""
Data Generation Module for E-commerce Semantic Search System
=============================================================

This module generates synthetic e-commerce product data with:
- Semantic variation (synonyms, paraphrases)
- Realistic descriptions (not just templates)
- Ground truth queries for evaluation

Why synthetic data with semantic variation matters:
- Tests whether vector search captures meaning beyond keywords
- Validates that synonyms like "sneakers" vs "running shoes" are matched
- Provides controlled ground truth for evaluation metrics
"""

import json
import random
import logging
from typing import List, Dict, Any, Tuple
from faker import Faker

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
Faker.seed(RANDOM_SEED)
fake = Faker()

# ============================================================================
# CATEGORY DEFINITIONS WITH SEMANTIC VARIATIONS
# ============================================================================

CATEGORIES = {
    "electronics": {
        "products": [
            # Each product has multiple semantic variations for titles
            {
                "base": "smartphone",
                "variations": ["mobile phone", "cell phone", "smart device", "handset"],
                "brands": ["Samsung", "Apple", "Google", "OnePlus", "Xiaomi", "Huawei"],
                "attributes": {"storage": ["64GB", "128GB", "256GB", "512GB"], "color": ["black", "white", "blue", "silver", "gold"]},
                "price_range": (299, 1299),
                "description_templates": [
                    "A powerful {variation} with {storage} storage and stunning {color} finish. Features advanced camera system and all-day battery life.",
                    "Experience the future with this {variation}. Packed with {storage} of space and a beautiful {color} design that stands out.",
                    "Premium {variation} featuring {storage} storage capacity. Available in elegant {color}. Perfect for professionals and tech enthusiasts.",
                    "Cutting-edge {variation} that redefines mobile technology. {storage} storage, {color} color, and flagship performance.",
                ]
            },
            {
                "base": "laptop",
                "variations": ["notebook", "portable computer", "ultrabook", "computing device"],
                "brands": ["Dell", "HP", "Lenovo", "ASUS", "Acer", "MSI", "Apple"],
                "attributes": {"ram": ["8GB", "16GB", "32GB"], "storage": ["256GB SSD", "512GB SSD", "1TB SSD"], "color": ["silver", "black", "space gray"]},
                "price_range": (599, 2499),
                "description_templates": [
                    "Powerful {variation} with {ram} RAM and {storage}. Sleek {color} chassis perfect for work and play.",
                    "High-performance {variation} featuring {ram} memory and {storage}. Lightweight {color} design for professionals on the go.",
                    "Premium {variation} engineered for productivity. Equipped with {ram} RAM, {storage}, and elegant {color} finish.",
                    "Versatile {variation} that handles any task. {ram} RAM ensures smooth multitasking, {storage} for all your files.",
                ]
            },
            {
                "base": "wireless earbuds",
                "variations": ["bluetooth earphones", "true wireless headphones", "wireless in-ear headphones", "cordless earbuds"],
                "brands": ["Apple", "Sony", "Bose", "Samsung", "Jabra", "Sennheiser"],
                "attributes": {"battery_life": ["6 hours", "8 hours", "10 hours"], "color": ["white", "black", "navy", "rose gold"]},
                "price_range": (49, 349),
                "description_templates": [
                    "Premium {variation} with {battery_life} battery life. Stunning {color} finish with active noise cancellation.",
                    "Immersive audio experience with these {variation}. {battery_life} playback time, available in stylish {color}.",
                    "Crystal clear sound from these {variation}. Enjoy {battery_life} of continuous listening in beautiful {color}.",
                    "High-fidelity {variation} designed for music lovers. {battery_life} battery, {color} color, superior comfort.",
                ]
            },
            {
                "base": "smartwatch",
                "variations": ["fitness tracker watch", "wearable device", "digital watch", "smart wristwatch"],
                "brands": ["Apple", "Samsung", "Garmin", "Fitbit", "Amazfit", "Huawei"],
                "attributes": {"display": ["AMOLED", "LCD", "Retina"], "color": ["black", "silver", "rose gold", "blue"]},
                "price_range": (99, 799),
                "description_templates": [
                    "Advanced {variation} with {display} display. Track your fitness goals in style with the {color} edition.",
                    "Stay connected with this sleek {variation}. Features {display} screen and comes in stunning {color}.",
                    "Monitor your health 24/7 with this {variation}. Vibrant {display} display, elegant {color} design.",
                    "The ultimate {variation} for active lifestyles. {display} technology, {color} case, water resistant.",
                ]
            },
            {
                "base": "tablet",
                "variations": ["digital slate", "portable touchscreen", "media tablet", "touch screen device"],
                "brands": ["Apple", "Samsung", "Microsoft", "Lenovo", "Amazon"],
                "attributes": {"screen_size": ["10 inch", "11 inch", "12.9 inch"], "storage": ["64GB", "128GB", "256GB"], "color": ["silver", "space gray", "black"]},
                "price_range": (299, 1299),
                "description_templates": [
                    "Versatile {variation} with {screen_size} display and {storage} storage. Available in {color} for work and entertainment.",
                    "Premium {variation} featuring {screen_size} screen. {storage} capacity in sleek {color} design.",
                    "Powerful {variation} with stunning {screen_size} display. Store everything with {storage}. Beautiful {color} finish.",
                    "Experience content like never before on this {variation}. {screen_size} screen, {storage}, {color} color.",
                ]
            },
        ]
    },
    "clothing": {
        "products": [
            {
                "base": "sneakers",
                "variations": ["running shoes", "athletic footwear", "sports shoes", "gym trainers", "workout shoes"],
                "brands": ["Nike", "Adidas", "Puma", "New Balance", "Reebok", "Under Armour"],
                "attributes": {"size": ["US 7", "US 8", "US 9", "US 10", "US 11", "US 12"], "color": ["white", "black", "blue", "red", "gray", "multicolor"]},
                "price_range": (59, 249),
                "description_templates": [
                    "Comfortable {variation} designed for peak performance. Available in {color}, size {size}. Lightweight and breathable.",
                    "Premium {variation} with advanced cushioning technology. {color} colorway, size {size}. Perfect for runners.",
                    "Stylish {variation} that transition from gym to street. {color} design, available in size {size}.",
                    "High-performance {variation} for athletes. Responsive {color} model in size {size}. Durable construction.",
                ]
            },
            {
                "base": "winter jacket",
                "variations": ["cold weather coat", "insulated parka", "winter puffer", "thermal outerwear", "warm coat"],
                "brands": ["North Face", "Columbia", "Patagonia", "Canada Goose", "Arc'teryx"],
                "attributes": {"size": ["S", "M", "L", "XL", "XXL"], "color": ["black", "navy", "olive", "burgundy", "gray"]},
                "price_range": (99, 899),
                "description_templates": [
                    "Stay warm with this premium {variation}. Size {size} in {color}. Waterproof and windproof for harsh conditions.",
                    "Cozy {variation} perfect for freezing temperatures. Available in {color}, size {size}. Down-filled for maximum warmth.",
                    "Durable {variation} built for outdoor adventures. {color} color, size {size}. Insulated hood and pockets.",
                    "Essential {variation} for winter weather. Stylish {color} design in size {size}. Rated for sub-zero temperatures.",
                ]
            },
            {
                "base": "jeans",
                "variations": ["denim pants", "blue jeans", "denim trousers", "casual pants"],
                "brands": ["Levi's", "Wrangler", "Calvin Klein", "Diesel", "True Religion", "Gap"],
                "attributes": {"size": ["28", "30", "32", "34", "36", "38"], "color": ["indigo", "black", "light wash", "dark wash", "vintage blue"], "fit": ["slim", "regular", "relaxed", "skinny"]},
                "price_range": (39, 199),
                "description_templates": [
                    "Classic {variation} with {fit} fit. Size {size} in {color}. Premium denim that lasts.",
                    "Comfortable {variation} perfect for everyday wear. {color} wash, size {size}, {fit} cut.",
                    "Stylish {variation} with modern {fit} silhouette. Available in {color}, size {size}.",
                    "Timeless {variation} crafted from quality denim. Size {size}, {color}, {fit} fit for all-day comfort.",
                ]
            },
            {
                "base": "t-shirt",
                "variations": ["casual tee", "cotton shirt", "crew neck top", "basic tee"],
                "brands": ["Hanes", "Gildan", "Champion", "Nike", "Adidas", "Uniqlo"],
                "attributes": {"size": ["XS", "S", "M", "L", "XL", "XXL"], "color": ["white", "black", "navy", "gray", "red", "green"]},
                "price_range": (12, 49),
                "description_templates": [
                    "Soft and comfortable {variation} in {color}. Size {size}. Perfect for layering or wearing alone.",
                    "Classic {variation} made from premium cotton. Available in {color}, size {size}. Breathable and durable.",
                    "Essential {variation} for your wardrobe. {color} color, size {size}. Machine washable and long-lasting.",
                    "Versatile {variation} that goes with everything. Size {size} in {color}. Comfortable fit all day.",
                ]
            },
            {
                "base": "dress",
                "variations": ["evening gown", "casual dress", "formal attire", "midi dress", "cocktail dress"],
                "brands": ["Zara", "H&M", "ASOS", "Nordstrom", "Mango", "Free People"],
                "attributes": {"size": ["XS", "S", "M", "L", "XL"], "color": ["black", "red", "navy", "floral", "burgundy", "emerald"]},
                "price_range": (29, 299),
                "description_templates": [
                    "Elegant {variation} perfect for special occasions. Size {size} in stunning {color}. Flattering silhouette.",
                    "Beautiful {variation} that makes a statement. Available in {color}, size {size}. Comfortable and chic.",
                    "Stunning {variation} for memorable moments. {color} design in size {size}. Premium fabric and finish.",
                    "Sophisticated {variation} that turns heads. Size {size}, {color}. Perfect fit and timeless style.",
                ]
            },
        ]
    },
    "home": {
        "products": [
            {
                "base": "coffee maker",
                "variations": ["coffee machine", "espresso maker", "coffee brewer", "automatic coffee system"],
                "brands": ["Keurig", "Nespresso", "Breville", "Mr. Coffee", "De'Longhi", "Cuisinart"],
                "attributes": {"capacity": ["single serve", "10 cup", "12 cup"], "color": ["black", "silver", "white", "red"]},
                "price_range": (29, 599),
                "description_templates": [
                    "Premium {variation} with {capacity} capacity. Sleek {color} finish. Brew your perfect cup every morning.",
                    "Advanced {variation} for coffee enthusiasts. {capacity}, available in {color}. Programmable and easy to clean.",
                    "Convenient {variation} with {capacity} brewing capacity. {color} design fits any kitchen. Quality coffee in minutes.",
                    "State-of-the-art {variation} that delivers barista-quality coffee. {capacity}, {color} finish, intuitive controls.",
                ]
            },
            {
                "base": "vacuum cleaner",
                "variations": ["floor cleaner", "dust collector", "home vacuum", "cleaning machine", "carpet cleaner"],
                "brands": ["Dyson", "Shark", "iRobot", "Bissell", "Hoover", "Miele"],
                "attributes": {"type": ["upright", "cordless", "robot", "canister"], "color": ["purple", "silver", "red", "black"]},
                "price_range": (99, 799),
                "description_templates": [
                    "Powerful {variation} with {type} design. {color} finish with advanced suction technology for all floor types.",
                    "Efficient {variation} that makes cleaning effortless. {type} model in {color}. HEPA filtration included.",
                    "Premium {variation} with {type} convenience. Available in {color}. Pet hair and allergen removal.",
                    "Next-generation {variation} featuring {type} technology. {color} color. Smart sensors and long battery life.",
                ]
            },
            {
                "base": "bedding set",
                "variations": ["duvet cover set", "comforter set", "sheet set", "bed linens"],
                "brands": ["Brooklinen", "Casper", "Parachute", "Amazon Basics", "Pottery Barn"],
                "attributes": {"size": ["Twin", "Full", "Queen", "King", "California King"], "color": ["white", "gray", "navy", "sage", "blush"], "material": ["cotton", "linen", "microfiber", "bamboo"]},
                "price_range": (49, 399),
                "description_templates": [
                    "Luxurious {variation} in {size} size. Beautiful {color} color in soft {material}. Transform your bedroom.",
                    "Premium {variation} made from quality {material}. {size} size, {color} color. Hotel-quality comfort at home.",
                    "Cozy {variation} perfect for restful sleep. {material} fabric in {color}. Available in {size}.",
                    "Elegant {variation} that elevates your bedroom. {size}, {color}, 100% {material}. Easy care and durable.",
                ]
            },
            {
                "base": "air purifier",
                "variations": ["air cleaner", "air filter system", "HEPA purifier", "air quality device"],
                "brands": ["Dyson", "Honeywell", "Levoit", "Coway", "Blueair", "Molekule"],
                "attributes": {"coverage": ["small room", "medium room", "large room", "whole house"], "color": ["white", "black", "silver"]},
                "price_range": (79, 699),
                "description_templates": [
                    "Advanced {variation} perfect for {coverage} coverage. {color} design removes 99.97% of airborne particles.",
                    "Quiet {variation} with HEPA filtration for {coverage}. Clean air in minutes. {color} finish.",
                    "Smart {variation} monitors and purifies air. {coverage} coverage in stylish {color}. App controlled.",
                    "Medical-grade {variation} for allergy relief. {coverage} capacity, {color} color. Ultra-quiet operation.",
                ]
            },
            {
                "base": "blender",
                "variations": ["food processor", "smoothie maker", "kitchen blender", "mixing machine"],
                "brands": ["Vitamix", "Ninja", "Blendtec", "KitchenAid", "NutriBullet", "Cuisinart"],
                "attributes": {"power": ["500W", "1000W", "1500W", "2000W"], "color": ["black", "silver", "red", "white"]},
                "price_range": (39, 599),
                "description_templates": [
                    "Powerful {variation} with {power} motor. {color} finish for any kitchen. Blend anything perfectly.",
                    "Professional-grade {variation} with {power} power. Available in {color}. Makes smoothies, soups, and more.",
                    "Versatile {variation} featuring {power} motor. {color} design with multiple speed settings and presets.",
                    "High-performance {variation} crushes ice and blends tough ingredients. {power}, {color}, easy to clean.",
                ]
            },
        ]
    },
    "sports": {
        "products": [
            {
                "base": "yoga mat",
                "variations": ["exercise mat", "fitness mat", "workout pad", "gym mat", "training mat"],
                "brands": ["Manduka", "Liforme", "Gaiam", "Lululemon", "JadeYoga", "Nike"],
                "attributes": {"thickness": ["3mm", "5mm", "6mm", "8mm"], "color": ["purple", "black", "blue", "teal", "pink", "green"]},
                "price_range": (19, 149),
                "description_templates": [
                    "Premium {variation} with {thickness} cushioning. Beautiful {color} color for comfortable practice.",
                    "Non-slip {variation} perfect for yoga and pilates. {thickness} thick in {color}. Eco-friendly materials.",
                    "Professional {variation} with superior grip. {color} design, {thickness} for joint protection.",
                    "Durable {variation} that supports every pose. {thickness} cushion, {color}. Easy to roll and carry.",
                ]
            },
            {
                "base": "dumbbell set",
                "variations": ["weight set", "hand weights", "free weights", "resistance weights", "strength training weights"],
                "brands": ["Bowflex", "CAP Barbell", "PowerBlock", "Amazon Basics", "NordicTrack"],
                "attributes": {"weight_range": ["5-25 lbs", "10-50 lbs", "5-52.5 lbs", "adjustable"], "color": ["black", "gray", "chrome", "neoprene coated"]},
                "price_range": (29, 599),
                "description_templates": [
                    "Versatile {variation} with {weight_range} range. {color} finish for home gym essential.",
                    "Complete {variation} for strength training. {weight_range}, {color}. Space-saving design.",
                    "Premium {variation} adjustable from {weight_range}. {color} with comfortable grip. Build muscle at home.",
                    "Professional-quality {variation} for serious training. {weight_range}, {color}. Durable construction.",
                ]
            },
            {
                "base": "treadmill",
                "variations": ["running machine", "jogging machine", "cardio trainer", "home runner", "folding treadmill"],
                "brands": ["NordicTrack", "Peloton", "ProForm", "Sole", "Horizon", "Bowflex"],
                "attributes": {"max_speed": ["10 mph", "12 mph", "15 mph"], "incline": ["10%", "12%", "15% auto"], "color": ["black", "silver", "gray"]},
                "price_range": (399, 2999),
                "description_templates": [
                    "Advanced {variation} with {max_speed} max speed and {incline} incline. {color} design with touchscreen.",
                    "Compact {variation} perfect for home workouts. Reaches {max_speed}, {incline} for varied training. {color}.",
                    "Professional {variation} with {max_speed} speed capability. {incline} adjustment, {color} frame. Quiet motor.",
                    "Feature-rich {variation} for serious runners. {max_speed}, {incline}, {color}. Built-in programs and tracking.",
                ]
            },
            {
                "base": "basketball",
                "variations": ["indoor basketball", "outdoor ball", "game ball", "basketball hoop ball", "street ball"],
                "brands": ["Spalding", "Wilson", "Nike", "Molten", "Baden", "Under Armour"],
                "attributes": {"size": ["Size 5 youth", "Size 6 women", "Size 7 official"], "material": ["leather", "composite", "rubber"]},
                "price_range": (19, 199),
                "description_templates": [
                    "Official {variation} with {material} construction. {size} for competitive play. Superior grip and durability.",
                    "Premium {variation} made from quality {material}. {size}. Perfect bounce and feel for any court.",
                    "Pro-level {variation} with {material} cover. {size}. Designed for indoor and outdoor performance.",
                    "High-quality {variation} for players of all levels. {size}, {material}. Long-lasting and game-ready.",
                ]
            },
            {
                "base": "resistance bands",
                "variations": ["exercise bands", "workout bands", "fitness bands", "stretch bands", "training bands"],
                "brands": ["TheraBand", "Fit Simplify", "Perform Better", "SPRI", "Bodylastics"],
                "attributes": {"resistance": ["light", "medium", "heavy", "extra heavy", "set of 5"], "color": ["multicolor", "black", "green", "blue"]},
                "price_range": (9, 79),
                "description_templates": [
                    "Versatile {variation} with {resistance} resistance. {color} set for full-body workouts anywhere.",
                    "Durable {variation} perfect for strength training. {resistance} level, {color}. Portable fitness solution.",
                    "Premium {variation} for rehabilitation and exercise. {resistance}, {color}. Safe and effective.",
                    "Complete {variation} kit with {resistance} levels. {color} bands for progressive training. Includes guide.",
                ]
            },
        ]
    },
    "beauty": {
        "products": [
            {
                "base": "moisturizer",
                "variations": ["face cream", "hydrating lotion", "skin cream", "daily moisturizing cream", "facial hydrator"],
                "brands": ["CeraVe", "Cetaphil", "Olay", "Neutrogena", "La Roche-Posay", "Clinique"],
                "attributes": {"skin_type": ["dry skin", "oily skin", "combination", "sensitive", "all skin types"], "size": ["1.7 oz", "3 oz", "4 oz"]},
                "price_range": (9, 89),
                "description_templates": [
                    "Nourishing {variation} formulated for {skin_type}. {size} bottle hydrates and protects all day.",
                    "Lightweight {variation} perfect for {skin_type}. {size} of gentle, effective moisture.",
                    "Dermatologist-recommended {variation} for {skin_type}. {size}. Non-greasy, fast-absorbing formula.",
                    "Premium {variation} that transforms {skin_type}. {size} jar with scientifically proven ingredients.",
                ]
            },
            {
                "base": "shampoo",
                "variations": ["hair cleanser", "hair wash", "scalp treatment", "cleansing shampoo"],
                "brands": ["Olaplex", "Kerastase", "Redken", "Paul Mitchell", "Moroccanoil", "Aussie"],
                "attributes": {"hair_type": ["damaged hair", "color-treated", "fine hair", "curly hair", "all hair types"], "size": ["8 oz", "12 oz", "16 oz"]},
                "price_range": (8, 49),
                "description_templates": [
                    "Salon-quality {variation} designed for {hair_type}. {size} bottle cleanses and nourishes deeply.",
                    "Gentle yet effective {variation} for {hair_type}. {size}. Leaves hair soft and manageable.",
                    "Professional {variation} repairs and strengthens {hair_type}. {size}. Sulfate-free formula.",
                    "Luxurious {variation} transforms {hair_type}. {size} of concentrated formula. Visible results.",
                ]
            },
            {
                "base": "lipstick",
                "variations": ["lip color", "lip tint", "lip stain", "matte lipstick", "liquid lip"],
                "brands": ["MAC", "NARS", "Charlotte Tilbury", "Fenty Beauty", "NYX", "Maybelline"],
                "attributes": {"finish": ["matte", "satin", "glossy", "velvet"], "color": ["red", "nude", "pink", "berry", "coral", "mauve"]},
                "price_range": (7, 39),
                "description_templates": [
                    "Stunning {variation} in {color} with {finish} finish. Long-wearing formula that feels comfortable.",
                    "Bold {variation} that makes a statement. {color} shade with {finish} texture. Highly pigmented.",
                    "Gorgeous {variation} in flattering {color}. {finish} finish. Moisturizing and buildable coverage.",
                    "Perfect everyday {variation} in versatile {color}. {finish} formula stays put all day.",
                ]
            },
            {
                "base": "perfume",
                "variations": ["fragrance", "eau de parfum", "cologne", "scent", "body mist"],
                "brands": ["Chanel", "Dior", "Tom Ford", "Jo Malone", "Yves Saint Laurent", "Gucci"],
                "attributes": {"scent_family": ["floral", "woody", "fresh", "oriental", "citrus"], "size": ["1 oz", "1.7 oz", "3.4 oz"]},
                "price_range": (29, 299),
                "description_templates": [
                    "Captivating {variation} with {scent_family} notes. {size} bottle for lasting impression.",
                    "Luxurious {variation} featuring {scent_family} accords. {size}. Sophisticated and memorable.",
                    "Signature {variation} with beautiful {scent_family} blend. {size}. Perfect for any occasion.",
                    "Elegant {variation} that defines your style. {scent_family} fragrance in {size}. Long-lasting.",
                ]
            },
            {
                "base": "sunscreen",
                "variations": ["sun protection", "SPF lotion", "UV protector", "sun block", "daily SPF"],
                "brands": ["La Roche-Posay", "Supergoop", "Neutrogena", "EltaMD", "CeraVe", "Coppertone"],
                "attributes": {"spf": ["SPF 30", "SPF 50", "SPF 70"], "type": ["mineral", "chemical", "hybrid"], "size": ["1.7 oz", "3 oz", "6 oz"]},
                "price_range": (9, 49),
                "description_templates": [
                    "Protective {variation} with {spf}. {type} formula in {size}. Broad spectrum UVA/UVB protection.",
                    "Lightweight {variation} featuring {spf}. {size} of {type} protection. No white cast.",
                    "Daily {variation} with {spf} protection. {type} sunscreen in {size}. Moisturizing formula.",
                    "Dermatologist-approved {variation} with {spf}. {size}, {type}. Water-resistant and gentle.",
                ]
            },
        ]
    }
}

# ============================================================================
# SEARCH QUERY DEFINITIONS WITH GROUND TRUTH
# ============================================================================

# These queries are designed to test different search scenarios:
# 1. Exact keyword matches (BM25 should work)
# 2. Synonym queries (Vector search should excel)
# 3. Natural language queries (Vector search advantage)
# 4. Multi-concept queries (Hybrid should work best)

QUERY_TEMPLATES = [
    # Synonym-heavy queries (BM25 will struggle, vector search should excel)
    {
        "query": "gym footwear for running",
        "target_products": ["sneakers"],
        "category": "clothing",
        "query_type": "synonym"
    },
    {
        "query": "mobile phone with good camera",
        "target_products": ["smartphone"],
        "category": "electronics",
        "query_type": "synonym"
    },
    {
        "query": "cordless audio buds for music",
        "target_products": ["wireless earbuds"],
        "category": "electronics",
        "query_type": "synonym"
    },
    {
        "query": "cold weather outerwear",
        "target_products": ["winter jacket"],
        "category": "clothing",
        "query_type": "synonym"
    },
    {
        "query": "hot beverage machine",
        "target_products": ["coffee maker"],
        "category": "home",
        "query_type": "synonym"
    },
    {
        "query": "floor cleaning device",
        "target_products": ["vacuum cleaner"],
        "category": "home",
        "query_type": "synonym"
    },
    {
        "query": "exercise pad for stretching",
        "target_products": ["yoga mat"],
        "category": "sports",
        "query_type": "synonym"
    },
    {
        "query": "face hydration cream",
        "target_products": ["moisturizer"],
        "category": "beauty",
        "query_type": "synonym"
    },
    {
        "query": "portable computing device",
        "target_products": ["laptop", "tablet"],
        "category": "electronics",
        "query_type": "synonym"
    },
    {
        "query": "strength training equipment",
        "target_products": ["dumbbell set", "resistance bands"],
        "category": "sports",
        "query_type": "synonym"
    },
    # Natural language queries
    {
        "query": "I need something to listen to podcasts while jogging",
        "target_products": ["wireless earbuds"],
        "category": "electronics",
        "query_type": "natural"
    },
    {
        "query": "what can I wear to stay warm this winter",
        "target_products": ["winter jacket"],
        "category": "clothing",
        "query_type": "natural"
    },
    {
        "query": "looking for comfortable shoes for my daily workout",
        "target_products": ["sneakers"],
        "category": "clothing",
        "query_type": "natural"
    },
    {
        "query": "need something to make my morning coffee quickly",
        "target_products": ["coffee maker"],
        "category": "home",
        "query_type": "natural"
    },
    {
        "query": "help me track my steps and heart rate",
        "target_products": ["smartwatch"],
        "category": "electronics",
        "query_type": "natural"
    },
    {
        "query": "I want to exercise at home without going to the gym",
        "target_products": ["yoga mat", "dumbbell set", "resistance bands", "treadmill"],
        "category": "sports",
        "query_type": "natural"
    },
    {
        "query": "something to protect my skin from the sun",
        "target_products": ["sunscreen"],
        "category": "beauty",
        "query_type": "natural"
    },
    {
        "query": "need a better night's sleep",
        "target_products": ["bedding set"],
        "category": "home",
        "query_type": "natural"
    },
    {
        "query": "what should I get for smoother, healthier hair",
        "target_products": ["shampoo"],
        "category": "beauty",
        "query_type": "natural"
    },
    {
        "query": "I want to start doing yoga at home",
        "target_products": ["yoga mat"],
        "category": "sports",
        "query_type": "natural"
    },
    # Short keyword queries (BM25 should work reasonably well)
    {
        "query": "laptop 16gb ram",
        "target_products": ["laptop"],
        "category": "electronics",
        "query_type": "keyword"
    },
    {
        "query": "black sneakers",
        "target_products": ["sneakers"],
        "category": "clothing",
        "query_type": "keyword"
    },
    {
        "query": "air purifier hepa",
        "target_products": ["air purifier"],
        "category": "home",
        "query_type": "keyword"
    },
    {
        "query": "matte lipstick red",
        "target_products": ["lipstick"],
        "category": "beauty",
        "query_type": "keyword"
    },
    {
        "query": "adjustable dumbbells",
        "target_products": ["dumbbell set"],
        "category": "sports",
        "query_type": "keyword"
    },
    # Multi-concept queries (testing hybrid search)
    {
        "query": "wireless noise cancelling headphones for commuting",
        "target_products": ["wireless earbuds"],
        "category": "electronics",
        "query_type": "multi_concept"
    },
    {
        "query": "professional blender for smoothies and soups",
        "target_products": ["blender"],
        "category": "home",
        "query_type": "multi_concept"
    },
    {
        "query": "elegant dress for evening party",
        "target_products": ["dress"],
        "category": "clothing",
        "query_type": "multi_concept"
    },
    {
        "query": "running machine with incline for cardio training",
        "target_products": ["treadmill"],
        "category": "sports",
        "query_type": "multi_concept"
    },
    {
        "query": "luxury fragrance woody notes gift",
        "target_products": ["perfume"],
        "category": "beauty",
        "query_type": "multi_concept"
    },
]


def generate_product(
    product_id: int,
    category: str,
    product_type: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a single product with realistic variation.
    
    Args:
        product_id: Unique identifier for the product
        category: Product category name
        product_type: Product type definition with variations and templates
    
    Returns:
        Dictionary containing product data
    """
    # Randomly choose variation for semantic diversity
    title_base = random.choice([product_type["base"]] + product_type["variations"])
    brand = random.choice(product_type["brands"])
    
    # Select random attributes
    attributes = {}
    template_vars = {}
    for attr_name, attr_values in product_type["attributes"].items():
        value = random.choice(attr_values)
        attributes[attr_name] = value
        template_vars[attr_name] = value
    
    template_vars["variation"] = title_base
    
    # Generate title with brand and key attribute
    key_attr = list(attributes.values())[0] if attributes else ""
    title = f"{brand} {title_base.title()} - {key_attr}".strip(" -")
    
    # Generate description from template
    description_template = random.choice(product_type["description_templates"])
    description = description_template.format(**template_vars)
    
    # Generate price
    min_price, max_price = product_type["price_range"]
    price = round(random.uniform(min_price, max_price), 2)
    
    # Generate rating (weighted towards higher ratings as typical for e-commerce)
    rating = round(random.triangular(3.0, 5.0, 4.5), 1)
    
    return {
        "id": f"prod_{product_id:05d}",
        "title": title,
        "description": description,
        "category": category,
        "price": price,
        "brand": brand,
        "rating": rating,
        "attributes": attributes,
        "product_type": product_type["base"]  # Store base type for query matching
    }


def generate_products(num_products: int = 1500) -> List[Dict[str, Any]]:
    """
    Generate synthetic e-commerce product dataset.
    
    This creates products with:
    - Semantic variations (synonyms in titles/descriptions)
    - Realistic pricing and ratings
    - Diverse attributes per category
    
    Args:
        num_products: Number of products to generate (default 1500)
    
    Returns:
        List of product dictionaries
    """
    logger.info(f"Generating {num_products} synthetic products...")
    
    products = []
    product_id = 1
    
    # Calculate products per category
    categories = list(CATEGORIES.keys())
    products_per_category = num_products // len(categories)
    
    for category_name, category_data in CATEGORIES.items():
        product_types = category_data["products"]
        products_per_type = products_per_category // len(product_types)
        
        for product_type in product_types:
            for _ in range(products_per_type):
                product = generate_product(product_id, category_name, product_type)
                products.append(product)
                product_id += 1
        
        # Fill remaining slots for this category
        remaining = products_per_category - (products_per_type * len(product_types))
        for _ in range(remaining):
            product_type = random.choice(product_types)
            product = generate_product(product_id, category_name, product_type)
            products.append(product)
            product_id += 1
    
    # Add extra products to reach target
    while len(products) < num_products:
        category_name = random.choice(categories)
        product_type = random.choice(CATEGORIES[category_name]["products"])
        product = generate_product(product_id, category_name, product_type)
        products.append(product)
        product_id += 1
    
    # Shuffle for variety
    random.shuffle(products)
    
    logger.info(f"Generated {len(products)} products across {len(categories)} categories")
    return products


def generate_queries(products: List[Dict[str, Any]], num_queries: int = 100) -> List[Dict[str, Any]]:
    """
    Generate search queries with ground truth relevance labels.
    
    Creates diverse query types:
    - Synonym queries (test semantic understanding)
    - Natural language queries (test conversational search)
    - Keyword queries (baseline for BM25)
    - Multi-concept queries (test hybrid search)
    
    Args:
        products: List of products to match queries against
        num_queries: Number of queries to generate
    
    Returns:
        List of query dictionaries with ground truth product IDs
    """
    logger.info(f"Generating {num_queries} search queries with ground truth...")
    
    queries = []
    
    # Create product lookup by type and category
    product_index = {}
    for product in products:
        key = (product["category"], product["product_type"])
        if key not in product_index:
            product_index[key] = []
        product_index[key].append(product["id"])
    
    query_id = 1
    
    # First, use predefined query templates
    for template in QUERY_TEMPLATES:
        relevant_ids = []
        for target_type in template["target_products"]:
            key = (template["category"], target_type)
            if key in product_index:
                # Get random subset of matching products as ground truth
                matching_products = product_index[key]
                # Use 5-15 products as relevant for each query
                sample_size = min(random.randint(5, 15), len(matching_products))
                relevant_ids.extend(random.sample(matching_products, sample_size))
        
        if relevant_ids:
            queries.append({
                "id": f"query_{query_id:03d}",
                "query": template["query"],
                "query_type": template["query_type"],
                "relevant_product_ids": relevant_ids
            })
            query_id += 1
    
    # Generate additional queries to reach target
    additional_templates = [
        # More synonym queries
        ("athletic shoes for training", "clothing", ["sneakers"], "synonym"),
        ("notebook computer for work", "electronics", ["laptop"], "synonym"),
        ("skin hydrating product", "beauty", ["moisturizer"], "synonym"),
        ("home exercise equipment", "sports", ["dumbbell set", "resistance bands", "yoga mat"], "synonym"),
        ("dust removal machine", "home", ["vacuum cleaner"], "synonym"),
        ("sleep comfort set", "home", ["bedding set"], "synonym"),
        ("wrist worn fitness device", "electronics", ["smartwatch"], "synonym"),
        ("touch screen portable device", "electronics", ["tablet"], "synonym"),
        ("lip color cosmetic", "beauty", ["lipstick"], "synonym"),
        ("aromatic spray fragrance", "beauty", ["perfume"], "synonym"),
        
        # More natural language queries
        ("best gift for a tech lover", "electronics", ["smartphone", "tablet", "wireless earbuds", "smartwatch"], "natural"),
        ("preparing for marathon training", "sports", ["sneakers", "treadmill"], "natural"),
        ("setting up my home office", "electronics", ["laptop", "tablet"], "natural"),
        ("getting ready for date night", "beauty", ["perfume", "lipstick"], "natural"),
        ("spring cleaning essentials", "home", ["vacuum cleaner"], "natural"),
        ("beach vacation must-haves", "beauty", ["sunscreen"], "natural"),
        ("morning routine products", "beauty", ["moisturizer", "shampoo"], "natural"),
        ("gifts for fitness enthusiast", "sports", ["yoga mat", "dumbbell set", "resistance bands"], "natural"),
        ("upgrading my kitchen", "home", ["coffee maker", "blender"], "natural"),
        ("starting a skincare routine", "beauty", ["moisturizer", "sunscreen"], "natural"),
        
        # More keyword queries  
        ("samsung smartphone 256gb", "electronics", ["smartphone"], "keyword"),
        ("nike running shoes", "clothing", ["sneakers"], "keyword"),
        ("dyson vacuum", "home", ["vacuum cleaner"], "keyword"),
        ("yoga mat 6mm", "sports", ["yoga mat"], "keyword"),
        ("chanel perfume", "beauty", ["perfume"], "keyword"),
        ("cotton bedding queen", "home", ["bedding set"], "keyword"),
        ("apple tablet", "electronics", ["tablet"], "keyword"),
        ("levis jeans slim", "clothing", ["jeans"], "keyword"),
        ("sony earbuds", "electronics", ["wireless earbuds"], "keyword"),
        ("vitamix blender", "home", ["blender"], "keyword"),
        
        # More multi-concept queries
        ("lightweight laptop for travel and meetings", "electronics", ["laptop"], "multi_concept"),
        ("comfortable jeans for everyday casual wear", "clothing", ["jeans"], "multi_concept"),
        ("quiet air purifier for bedroom allergies", "home", ["air purifier"], "multi_concept"),
        ("outdoor basketball for street games", "sports", ["basketball"], "multi_concept"),
        ("gentle shampoo for color treated hair", "beauty", ["shampoo"], "multi_concept"),
        ("folding treadmill for small apartment", "sports", ["treadmill"], "multi_concept"),
        ("formal dress black elegant", "clothing", ["dress"], "multi_concept"),
        ("sensitive skin facial moisturizer", "beauty", ["moisturizer"], "multi_concept"),
        ("robot vacuum pet hair", "home", ["vacuum cleaner"], "multi_concept"),
        ("premium espresso machine home barista", "home", ["coffee maker"], "multi_concept"),
    ]
    
    for query_text, category, target_types, query_type in additional_templates:
        if query_id > num_queries:
            break
            
        relevant_ids = []
        for target_type in target_types:
            key = (category, target_type)
            if key in product_index:
                matching_products = product_index[key]
                sample_size = min(random.randint(5, 15), len(matching_products))
                relevant_ids.extend(random.sample(matching_products, sample_size))
        
        if relevant_ids:
            queries.append({
                "id": f"query_{query_id:03d}",
                "query": query_text,
                "query_type": query_type,
                "relevant_product_ids": relevant_ids
            })
            query_id += 1
    
    # Add category-specific variation queries to reach target
    while len(queries) < num_queries:
        category = random.choice(list(CATEGORIES.keys()))
        product_type = random.choice(CATEGORIES[category]["products"])
        
        # Generate query using product variation
        variation = random.choice(product_type["variations"])
        brand = random.choice(product_type["brands"])
        
        query_style = random.choice(["brand_variation", "variation_only", "attribute_query"])
        
        if query_style == "brand_variation":
            query_text = f"{brand} {variation}"
        elif query_style == "variation_only":
            query_text = f"best {variation}"
        else:
            attr = random.choice(list(product_type["attributes"].keys()))
            attr_val = random.choice(product_type["attributes"][attr])
            query_text = f"{variation} {attr_val}"
        
        key = (category, product_type["base"])
        if key in product_index:
            matching_products = product_index[key]
            sample_size = min(random.randint(5, 15), len(matching_products))
            relevant_ids = random.sample(matching_products, sample_size)
            
            queries.append({
                "id": f"query_{query_id:03d}",
                "query": query_text,
                "query_type": "generated",
                "relevant_product_ids": relevant_ids
            })
            query_id += 1
    
    logger.info(f"Generated {len(queries)} queries")
    
    # Log query type distribution
    type_counts = {}
    for q in queries:
        qtype = q["query_type"]
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
    logger.info(f"Query type distribution: {type_counts}")
    
    return queries


def save_data(products: List[Dict[str, Any]], queries: List[Dict[str, Any]], output_dir: str = ".") -> Tuple[str, str]:
    """
    Save generated data to JSON files.
    
    Args:
        products: List of product dictionaries
        queries: List of query dictionaries
        output_dir: Directory to save files (default current directory)
    
    Returns:
        Tuple of (products_path, queries_path)
    """
    import os
    
    products_path = os.path.join(output_dir, "products.json")
    queries_path = os.path.join(output_dir, "queries.json")
    
    with open(products_path, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(products)} products to {products_path}")
    
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(queries)} queries to {queries_path}")
    
    return products_path, queries_path


def main():
    """Main function to generate and save synthetic data."""
    logger.info("=" * 60)
    logger.info("E-commerce Semantic Search - Data Generation")
    logger.info("=" * 60)
    
    # Generate products
    products = generate_products(num_products=1500)
    
    # Generate queries with ground truth
    queries = generate_queries(products, num_queries=100)
    
    # Save to files
    save_data(products, queries)
    
    # Print sample data
    logger.info("\n" + "=" * 60)
    logger.info("Sample Products:")
    logger.info("=" * 60)
    for product in products[:3]:
        print(json.dumps(product, indent=2))
        print()
    
    logger.info("\n" + "=" * 60)
    logger.info("Sample Queries:")
    logger.info("=" * 60)
    for query in queries[:3]:
        print(json.dumps(query, indent=2))
        print()
    
    logger.info("\nData generation complete!")


if __name__ == "__main__":
    main()
