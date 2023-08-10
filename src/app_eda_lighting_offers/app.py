import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from typing import Any, List
import plotly.express as px

@st.cache_data
def load_input_data(input_filename: str) -> pd.DataFrame:
	df = pd.read_csv(input_filename)
	df['OFFER_DURATION'] = df['OFFER_DURATION']/3600
	df = df.round({'OFFER_DURATION':0})
	return df

@st.cache_data
def generate_filter_lists(df_offers: pd.DataFrame,
						  field_name: str,
						  default_name: str) -> List:
	list_values = list(df_offers[field_name].unique())
	list_values.insert(0, default_name)
	return list_values

@st.cache_data
def get_pie_1(df_offers: pd.DataFrame) -> Any:
	df_quantities = df_offers.groupby(['DOM_DOMAIN_AGG1'])\
								[['SOLD_QUANTITY',
								  'INVOLVED_STOCK',
								  'SOLD_AMOUNT']].sum().reset_index()
	fig = px.pie(df_quantities,
				 values='SOLD_QUANTITY',
				 names='DOM_DOMAIN_AGG1',
				 title='Total units sold by category'
				)
	fig.update_traces(textposition='inside')
	fig.update_layout(uniformtext_minsize=12,
					  uniformtext_mode='hide',
					  showlegend=True
					 )
	return fig

@st.cache_data
def get_timeseries(df_offers: pd.DataFrame) -> Any:
	df_times = df_offers.groupby('OFFER_START_DATE')\
						[['INVOLVED_STOCK', 'SOLD_QUANTITY']]\
						.sum().reset_index()
	df_times = pd.melt(df_times,
						id_vars=['OFFER_START_DATE'],
						value_vars=['INVOLVED_STOCK', 'SOLD_QUANTITY'],
						var_name='VARIABLE',
						value_name='VALUE')
	fig = px.line(df_times,
				  x="OFFER_START_DATE",
				  y="VALUE",
				  title='Number units offered and sold',
				  color='VARIABLE', symbol='VARIABLE')
	return fig

@st.cache_data
def get_scatter_1(df_offers: pd.DataFrame) -> Any:
	fig = px.scatter(df_offers,
					 x="SOLD_AMOUNT",
					 y="OFFER_DURATION",
					 title='Relationship between price and timespan',
					 color="DOM_DOMAIN_AGG1",
                 	 size='SOLD_QUANTITY',
                 	 hover_data=['SOLD_QUANTITY'])
	fig.update_layout(showlegend=True)
	return fig

@st.cache_data
def get_bar_1(df_offers: pd.DataFrame) -> Any:
	df_revenue = df_offers.groupby('DOMAIN_ID')['SOLD_AMOUNT']\
						  .sum().reset_index()\
						  .sort_values('SOLD_AMOUNT', ascending=False)\
						  .reset_index()
	df_revenue = df_revenue[:20].iloc[::-1]
	fig = px.bar(df_revenue,
				 x='SOLD_AMOUNT',
				 y='DOMAIN_ID',
				 title='Top products by revenue')
	return fig


@st.cache_data
def get_violin_1(df_offers: pd.DataFrame) -> Any:

	fig = px.violin(df_offers,
					y="SOLD_AMOUNT",
					x="VERTICAL",
					color="DOM_DOMAIN_AGG1",
					box=True,
					title='Distribution of price by category')
	fig.update_layout(showlegend=False)
	return fig



input_filename = 'data/ofertas_relampago_preproc.csv'
image_filename = 'data/puppy_delivery.jpg'
df_offers = load_input_data(input_filename)
list_verticals = generate_filter_lists(df_offers, 'VERTICAL', 'ALL VERTICALS')
list_categories = generate_filter_lists(df_offers, 'DOM_DOMAIN_AGG1',
										'ALL CATEGORIES')
list_products = generate_filter_lists(df_offers, 'DOMAIN_ID', 'ALL PRODUCTS')
list_shipping = generate_filter_lists(df_offers, 'SHIPPING_PAYMENT_TYPE',
									  'ALL SHIPPING TYPES')


df_filtered = df_offers.copy()

# Filter data

select_vertical = st.sidebar.selectbox('VERTICAL', list_verticals, index=0)

if select_vertical != 'ALL VERTICALS':
	df_filtered = df_filtered[df_filtered['VERTICAL']==select_vertical]


select_category = st.sidebar.selectbox('CATEGORY', list_categories, index=0)

if select_category != 'ALL CATEGORIES':
	df_filtered = df_filtered[df_filtered['DOM_DOMAIN_AGG1']==select_category]


select_product = st.sidebar.selectbox('PRODUCT', list_products, index=0)

if select_product != 'ALL PRODUCTS':
	df_filtered = df_filtered[df_filtered['DOMAIN_ID']==select_product]


select_shipping = st.sidebar.selectbox('SHIPPING TYPE', list_shipping, index=0)

if select_shipping != 'ALL SHIPPING TYPES':
	df_filtered = df_filtered[df_filtered['SHIPPING_PAYMENT_TYPE']==\
																select_shipping]



st.sidebar.image(image_filename)


st.title(':zap: Lighting Offers :zap:')

timeseries_1 = get_timeseries(df_filtered)
st.plotly_chart(timeseries_1, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
	pie_1 = get_pie_1(df_filtered)
	st.plotly_chart(pie_1, use_container_width=True)

with col2:
	bar_1 = get_bar_1(df_filtered)
	st.plotly_chart(bar_1, use_container_width=True)

scatter_1 = get_scatter_1(df_filtered)
st.plotly_chart(scatter_1, use_container_width=True)


violin_1 = get_violin_1(df_filtered)
st.plotly_chart(violin_1, use_container_width=True)

