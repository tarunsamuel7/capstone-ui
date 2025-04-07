
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import random

# Import your custom modules
from DDPGPortfolioRecommender import DDPGPortfolioRecommender
from utils.recommender_utils import (
    simulate_future_recommendations_with_realistic_profits,
    display_date_based_recommendations,
    export_date_based_recommendations,
    plot_trading_summary,
)

# Add methods to the class
DDPGPortfolioRecommender.simulate_future_recommendations_with_realistic_profits = (
    simulate_future_recommendations_with_realistic_profits
)
DDPGPortfolioRecommender.display_date_based_recommendations = (
    display_date_based_recommendations
)
DDPGPortfolioRecommender.export_date_based_recommendations = (
    export_date_based_recommendations
)
DDPGPortfolioRecommender.plot_trading_summary = (
    plot_trading_summary
)

def main():
    st.title("DDPG Portfolio Recommender")
    
    # Set paths to your model and data
    model_path = "data/ddpg_portfolio_model.zip"
    data_path = "data/historical_data.csv"
    
    # User input for investment amount
    st.header("Investment Settings")
    investment_amount = st.number_input(
        "Investment Amount ($)",
        min_value=1000,
        max_value=100000,
        value=10000,
        step=1000,
    )
    
    # Button to run the simulation
    if st.button("Run Portfolio Simulation"):
        with st.spinner("Running simulation..."):
            try:
                # Create the recommender
                recommender = DDPGPortfolioRecommender(
                    model_path=model_path,
                    data_path=data_path,
                    max_stocks=100,
                    lookback=30,
                    feature_dimension=7,  # Adjust based on your model
                )
                
                # Run the simulation
                date_recommendations, final_value, realized_profit = (
                    recommender.simulate_future_recommendations_with_realistic_profits(
                        amount_cad=investment_amount,
                        future_days=30,
                        price_change_range=(-0.15, 0.25),
                    )
                )
                
                # Display the final value (as requested)
                st.success("Simulation completed successfully!")
                st.header("Results")
                st.metric(
                    label="Final Portfolio Value",
                    value=f"${final_value:.2f}",
                    delta=f"{(final_value/investment_amount - 1)*100:.2f}%"
                )
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("Please check that your model and data paths are correct.")

if __name__ == "__main__":
    main()
