"""
Módulo de funções ETL para processamento de dados de e-commerce.

Este módulo contém funções para extração, transformação e carga de dados
do dataset de e-commerce da Olist para uso no Power BI.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def extract_data(base_path='../data/raw/'):
    """
    Extrai dados dos arquivos CSV do dataset de e-commerce.
    
    Parameters:
    -----------
    base_path : str
        Caminho base para os arquivos de dados
        
    Returns:
    --------
    dict
        Dicionário contendo os dataframes carregados
    """
    try:
        datasets = {
            'customers': pd.read_csv(f'{base_path}olist_customers_dataset.csv'),
            'orders': pd.read_csv(f'{base_path}olist_orders_dataset.csv'),
            'order_items': pd.read_csv(f'{base_path}olist_order_items_dataset.csv'),
            'products': pd.read_csv(f'{base_path}olist_products_dataset.csv'),
            'sellers': pd.read_csv(f'{base_path}olist_sellers_dataset.csv'),
            'reviews': pd.read_csv(f'{base_path}olist_order_reviews_dataset.csv')
        }
        
        # Carregando tradução de categorias se existir
        if os.path.exists(f'{base_path}product_category_name_translation.csv'):
            datasets['category_translation'] = pd.read_csv(f'{base_path}product_category_name_translation.csv')
        
        print("Datasets extraídos com sucesso!")
        return datasets
    
    except FileNotFoundError as e:
        print(f"Erro ao extrair datasets: {e}")
        return None


def transform_data(raw_data):
    """
    Transforma os dados brutos para preparação para análise.
    
    Parameters:
    -----------
    raw_data : dict
        Dicionário contendo os dataframes brutos
        
    Returns:
    --------
    dict
        Dicionário contendo os dataframes transformados
    """
    transformed_data = {}
    
    # Copiando os dataframes para não modificar os originais
    for key, df in raw_data.items():
        transformed_data[key] = df.copy()
    
    # Convertendo colunas de data para datetime
    date_columns = {
        'orders': ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
                  'order_delivered_customer_date', 'order_estimated_delivery_date'],
        'reviews': ['review_creation_date', 'review_answer_timestamp'],
        'order_items': ['shipping_limit_date']
    }
    
    for table, columns in date_columns.items():
        if table in transformed_data:
            for col in columns:
                if col in transformed_data[table].columns:
                    transformed_data[table][col] = pd.to_datetime(transformed_data[table][col], errors='coerce')
    
    # Tratando valores ausentes
    for key, df in transformed_data.items():
        # Para colunas numéricas, preenchemos com a mediana
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
        
        # Para colunas de texto, preenchemos com 'unknown'
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna('unknown')
    
    # Adicionando colunas derivadas úteis
    if 'orders' in transformed_data:
        # Extraindo componentes de data
        transformed_data['orders']['purchase_year'] = transformed_data['orders']['order_purchase_timestamp'].dt.year
        transformed_data['orders']['purchase_month'] = transformed_data['orders']['order_purchase_timestamp'].dt.month
        transformed_data['orders']['purchase_day'] = transformed_data['orders']['order_purchase_timestamp'].dt.day
        transformed_data['orders']['purchase_dayofweek'] = transformed_data['orders']['order_purchase_timestamp'].dt.dayofweek
        transformed_data['orders']['purchase_quarter'] = transformed_data['orders']['order_purchase_timestamp'].dt.quarter
        
        # Calculando tempo de entrega (em dias)
        transformed_data['orders']['delivery_time_days'] = (
            transformed_data['orders']['order_delivered_customer_date'] - 
            transformed_data['orders']['order_purchase_timestamp']
        ).dt.total_seconds() / (24 * 3600)
        
        # Calculando atraso na entrega (em dias, negativo significa entrega antecipada)
        transformed_data['orders']['delivery_delay_days'] = (
            transformed_data['orders']['order_delivered_customer_date'] - 
            transformed_data['orders']['order_estimated_delivery_date']
        ).dt.total_seconds() / (24 * 3600)
        
        # Marcando se entrega foi feita dentro do prazo
        transformed_data['orders']['delivered_on_time'] = transformed_data['orders']['delivery_delay_days'] <= 0
    
    # Traduzindo categorias de produtos se a tabela de tradução estiver disponível
    if 'products' in transformed_data and 'category_translation' in transformed_data:
        transformed_data['products'] = pd.merge(
            transformed_data['products'],
            transformed_data['category_translation'],
            on='product_category_name',
            how='left'
        )
    
    return transformed_data


def create_dimensional_model(transformed_data):
    """
    Cria um modelo dimensional (estrela) a partir dos dados transformados.
    
    Parameters:
    -----------
    transformed_data : dict
        Dicionário contendo os dataframes transformados
        
    Returns:
    --------
    tuple
        (dim_tables, fact_table) - tabelas dimensionais e tabela fato
    """
    dim_tables = {}
    
    # Dimensão Data
    if 'orders' in transformed_data:
        # Encontrando o intervalo de datas
        min_date = transformed_data['orders']['order_purchase_timestamp'].min()
        max_date = transformed_data['orders']['order_purchase_timestamp'].max()
        
        if pd.notna(min_date) and pd.notna(max_date):
            # Criando sequência de datas
            date_range = pd.date_range(start=min_date, end=max_date, freq='D')
            
            # Criando dimensão de data
            dim_date = pd.DataFrame({
                'date': date_range,
                'year': date_range.year,
                'month': date_range.month,
                'day': date_range.day,
                'dayofweek': date_range.dayofweek,
                'quarter': date_range.quarter,
                'is_weekend': date_range.dayofweek.isin([5, 6]).astype(int),
                'month_name': date_range.strftime('%B'),
                'dayofweek_name': date_range.strftime('%A')
            })
            
            # Adicionando chave primária
            dim_date['id'] = dim_date['date'].dt.strftime('%Y%m%d').astype(int)
            
            dim_tables['date'] = dim_date
    
    # Dimensão Cliente
    if 'customers' in transformed_data:
        dim_customer = transformed_data['customers'].copy()
        dim_customer['id'] = dim_customer['customer_id']
        dim_tables['customer'] = dim_customer
    
    # Dimensão Produto
    if 'products' in transformed_data:
        dim_product = transformed_data['products'].copy()
        dim_product['id'] = dim_product['product_id']
        
        # Adicionando nome da categoria em inglês se disponível
        if 'product_category_name_english' not in dim_product.columns:
            dim_product['product_category_name_english'] = dim_product['product_category_name']
        
        dim_tables['product'] = dim_product
    
    # Dimensão Vendedor
    if 'sellers' in transformed_data:
        dim_seller = transformed_data['sellers'].copy()
        dim_seller['id'] = dim_seller['seller_id']
        dim_tables['seller'] = dim_seller
    
    # Dimensão Pedido
    if 'orders' in transformed_data:
        dim_order = transformed_data['orders'][['order_id', 'order_status', 'order_purchase_timestamp', 
                                              'order_approved_at', 'order_delivered_carrier_date',
                                              'order_delivered_customer_date', 'order_estimated_delivery_date',
                                              'delivery_time_days', 'delivery_delay_days', 'delivered_on_time']].copy()
        dim_order['id'] = dim_order['order_id']
        dim_tables['order'] = dim_order
    
    # Dimensão Avaliação
    if 'reviews' in transformed_data:
        dim_review = transformed_data['reviews'].copy()
        dim_review['id'] = dim_review['review_id']
        dim_tables['review'] = dim_review
    
    # Tabela Fato - Vendas
    if all(k in transformed_data for k in ['orders', 'order_items']):
        # Juntando pedidos e itens
        fact_sales = pd.merge(
            transformed_data['order_items'],
            transformed_data['orders'][['order_id', 'customer_id', 'order_purchase_timestamp']],
            on='order_id',
            how='inner'
        )
        
        # Adicionando chave para dimensão de data
        fact_sales['date_id'] = fact_sales['order_purchase_timestamp'].dt.strftime('%Y%m%d').astype(int)
        
        # Selecionando colunas relevantes
        fact_sales = fact_sales[['order_id', 'order_item_id', 'product_id', 'seller_id', 
                                'customer_id', 'date_id', 'price', 'freight_value']]
        
        # Adicionando avaliações se disponíveis
        if 'reviews' in transformed_data:
            reviews_simple = transformed_data['reviews'][['order_id', 'review_score']].copy()
            fact_sales = pd.merge(fact_sales, reviews_simple, on='order_id', how='left')
            fact_sales['review_score'] = fact_sales['review_score'].fillna(0).astype(int)
    else:
        fact_sales = pd.DataFrame()
    
    return dim_tables, fact_sales


def create_aggregated_tables(fact_table, dim_tables):
    """
    Cria tabelas agregadas para análise a partir da tabela fato e dimensões.
    
    Parameters:
    -----------
    fact_table : pandas.DataFrame
        Tabela fato
    dim_tables : dict
        Dicionário contendo as tabelas dimensionais
        
    Returns:
    --------
    dict
        Dicionário contendo as tabelas agregadas
    """
    agg_tables = {}
    
    # Vendas por data
    if 'date' in dim_tables:
        # Agrupando vendas por data_id
        sales_by_date_id = fact_table.groupby('date_id').agg({
            'order_id': 'nunique',
            'price': 'sum',
            'freight_value': 'sum'
        }).reset_index()
        
        sales_by_date_id.columns = ['date_id', 'order_count', 'total_sales', 'total_freight']
        
        # Juntando com dimensão de data para obter informações temporais
        sales_by_date = pd.merge(
            sales_by_date_id,
            dim_tables['date'][['id', 'year', 'month', 'quarter']],
            left_on='date_id',
            right_on='id',
            how='inner'
        )
        
        # Agregando por mês
        sales_by_month = sales_by_date.groupby(['year', 'month', 'quarter']).agg({
            'order_count': 'sum',
            'total_sales': 'sum',
            'total_freight': 'sum'
        }).reset_index()
        
        # Calculando métricas adicionais
        sales_by_month['avg_order_value'] = sales_by_month['total_sales'] / sales_by_month['order_count']
        sales_by_month['freight_percentage'] = (sales_by_month['total_freight'] / sales_by_month['total_sales']) * 100
        
        agg_tables['sales_by_date'] = sales_by_month
    
    # Vendas por categoria de produto
    if 'product' in dim_tables:
        # Juntando tabela fato com dimensão de produto
        sales_with_product = pd.merge(
            fact_table,
            dim_tables['product'][['id', 'product_category_name', 'product_category_name_english']],
            left_on='product_id',
            right_on='id',
            how='inner'
        )
        
        # Agrupando por categoria
        category_name_col = 'product_category_name_english' if 'product_category_name_english' in dim_tables['product'].columns else 'product_category_name'
        sales_by_category = sales_with_product.groupby(category_name_col).agg({
            'order_id': 'nunique',
            'price': 'sum',
            'freight_value': 'sum'
        }).reset_index()
        
        sales_by_category.columns = ['category_name', 'order_count', 'total_sales', 'total_freight']
        sales_by_category['avg_order_value'] = sales_by_category['total_sales'] / sales_by_category['order_count']
        
        agg_tables['sales_by_category'] = sales_by_category
    
    # Vendas por localização (estado)
    if 'customer' in dim_tables:
        # Juntando tabela fato com dimensão de cliente
        sales_with_customer = pd.merge(
            fact_table,
            dim_tables['customer'][['id', 'customer_state', 'customer_city']],
            left_on='customer_id',
            right_on='id',
            how='inner'
        )
        
        # Agrupando por estado
        sales_by_state = sales_with_customer.groupby('customer_state').agg({
            'order_id': 'nunique',
            'price': 'sum',
            'freight_value': 'sum'
        }).reset_index()
        
        sales_by_state.columns = ['state', 'order_count', 'total_sales', 'total_freight']
        sales_by_state['avg_order_value'] = sales_by_state['total_sales'] / sales_by_state['order_count']
        
        # Agrupando por cidade (top cidades)
        sales_by_city = sales_with_customer.groupby(['customer_state', 'customer_city']).agg({
            'order_id': 'nunique',
            'price': 'sum'
        }).reset_index()
        
        sales_by_city.columns = ['state', 'city', 'order_count', 'total_sales']
        sales_by_city['location'] = sales_by_city['city'] + ' (' + sales_by_city['state'] + ')'
        
        agg_tables['sales_by_location'] = sales_by_state
        agg_tables['sales_by_city'] = sales_by_city
    
    # Vendas por vendedor
    if 'seller' in dim_tables:
        # Juntando tabela fato com dimensão de vendedor
        sales_with_seller = pd.merge(
            fact_table,
            dim_tables['seller'][['id', 'seller_state', 'seller_city']],
            left_on='seller_id',
            right_on='id',
            how='inner'
        )
        
        # Agrupando por vendedor
        sales_by_seller = sales_with_seller.groupby('seller_id').agg({
            'order_id': 'nunique',
            'price': 'sum',
            'freight_value': 'sum'
        }).reset_index()
        
        sales_by_seller.columns = ['seller_id', 'order_count', 'total_sales', 'total_freight']
        sales_by_seller['avg_order_value'] = sales_by_seller['total_sales'] / sales_by_seller['order_count']
        
        agg_tables['sales_by_seller'] = sales_by_seller
    
    # Métricas de avaliação
    if 'review_score' in fact_table.columns:
        # Agrupando por pontuação de avaliação
        review_metrics = fact_table.groupby('review_score').agg({
            'order_id': 'nunique',
            'price': 'sum'
        }).reset_index()
        
        review_metrics.columns = ['review_score', 'order_count', 'total_sales']
        
        # Calculando NPS
        if not review_metrics.empty:
            total_reviews = review_metrics['order_count'].sum()
            promoters = review_metrics[review_metrics['review_score'] == 5]['order_count'].sum()
            detractors = review_metrics[review_metrics['review_score'] <= 3]['order_count'].sum()
            
            nps = (promoters / total_reviews * 100) - (detractors / total_reviews * 100)
            
            review_metrics['nps'] = nps
        
        agg_tables['review_metrics'] = review_metrics
    
    return agg_tables


def export_to_power_bi(dim_tables, fact_table, agg_tables, output_path='../data/transformed/'):
    """
    Exporta as tabelas dimensionais, fato e agregadas para uso no Power BI.
    
    Parameters:
    -----------
    dim_tables : dict
        Dicionário contendo as tabelas dimensionais
    fact_table : pandas.DataFrame
        Tabela fato
    agg_tables : dict
        Dicionário contendo as tabelas agregadas
    output_path : str
        Caminho para salvar os arquivos
        
    Returns:
    --------
    bool
        True se a exportação foi bem-sucedida, False caso contrário
    """
    try:
        # Criando diretório se não existir
        os.makedirs(output_path, exist_ok=True)
        
        # Salvando tabelas dimensionais
        for name, df in dim_tables.items():
            df.to_csv(f'{output_path}dim_{name}.csv', index=False)
            df.to_parquet(f'{output_path}dim_{name}.parquet')
        
        # Salvando tabela fato
        fact_table.to_csv(f'{output_path}fact_sales.csv', index=False)
        fact_table.to_parquet(f'{output_path}fact_sales.parquet')
        
        # Salvando tabelas agregadas
        for name, df in agg_tables.items():
            df.to_csv(f'{output_path}agg_{name}.csv', index=False)
            df.to_parquet(f'{output_path}agg_{name}.parquet')
        
        print(f"Dados exportados com sucesso para {output_path}")
        return True
    
    except Exception as e:
        print(f"Erro ao exportar dados: {e}")
        return False


def create_power_bi_instructions(output_path='../reports/dashboard/'):
    """
    Cria um arquivo de instruções para integração com Power BI.
    
    Parameters:
    -----------
    output_path : str
        Caminho para salvar o arquivo de instruções
        
    Returns:
    --------
    bool
        True se a criação foi bem-sucedida, False caso contrário
    """
    try:
        # Criando diretório se não existir
        os.makedirs(output_path, exist_ok=True)
        
        # Conteúdo das instruções
        instructions = """# Instruções para Integração com Power BI

## 1. Importação de Dados

### Opção 1: Importar arquivos Parquet (Recomendado)
1. Abra o Power BI Desktop
2. Clique em "Obter Dados" > "Mais..." > "Arquivo" > "Parquet"
3. Navegue até a pasta `data/transformed`
4. Selecione os arquivos .parquet:
   - fact_sales.parquet
   - dim_customer.parquet
   - dim_product.parquet
   - dim_seller.parquet
   - dim_date.parquet
   - dim_order.parquet
   - dim_review.parquet

### Opção 2: Importar arquivos CSV
1. Abra o Power BI Desktop
2. Clique em "Obter Dados" > "Texto/CSV"
3. Navegue até a pasta `data/transformed`
4. Selecione os arquivos .csv (mesmos nomes acima, com extensão .csv)

## 2. Configuração do Modelo de Dados

### Configurar Relações
1. Vá para a visualização "Modelo" (ícone de diagrama no lado esquerdo)
2. Crie as seguintes relações:
   - fact_sales[customer_id] → dim_customer[id]
   - fact_sales[product_id] → dim_product[id]
   - fact_sales[seller_id] → dim_seller[id]
   - fact_sales[date_id] → dim_date[id]
   - fact_sales[order_id] → dim_order[id]

### Criar Medidas Calculadas
1. Clique com o botão direito na tabela fact_sales > "Nova medida"
2. Crie as seguintes medidas:

```
Total Vendas = SUM(fact_sales[price])
Total Frete = SUM(fact_sales[freight_value])
Número de Pedidos = DISTINCTCOUNT(fact_sales[order_id])
Ticket Médio = DIVIDE([Total Vendas], [Número de Pedidos])
Percentual de Frete = DIVIDE([Total Frete], [Total Vendas]) * 100
```

## 3. Criação do Dashboard

### Página 1: Visão Geral
1. Adicione cartões com as principais métricas:
   - Total de Vendas
   - Número de Pedidos
   - Ticket Médio
   - Percentual de Frete
2. Adicione um gráfico de linhas para mostrar a tendência de vendas ao longo do tempo
3. Adicione um gráfico de barras para as top 10 categorias de produtos
4. Adicione um mapa para mostrar vendas por estado

### Página 2: Análise de Produtos
1. Adicione uma tabela com as categorias de produtos e suas métricas
2. Adicione um gráfico de dispersão relacionando preço e frete
3. Adicione um gráfico de barras para os produtos mais vendidos

### Página 3: Análise de Clientes
1. Adicione um mapa de calor de vendas por estado e cidade
2. Adicione um gráfico de pizza para distribuição de avaliações
3. Adicione um gráfico de barras para relação entre avaliação e tempo de entrega

## 4. Adicionar Segmentações de Dados
1. Adicione segmentações para:
   - Período (ano, trimestre, mês)
   - Categoria de produto
   - Estado do cliente
   - Faixa de preço

## 5. Formatação e Finalização
1. Aplique um tema consistente (Arquivo > Opções e configurações > Mudar tema)
2. Adicione título e descrições a cada visualização
3. Organize as visualizações de forma lógica e atraente
4. Adicione botões de navegação entre páginas

## 6. Salvar e Compartilhar
1. Salve o arquivo .pbix na pasta `reports/dashboard`
2. Para compartilhar, você pode:
   - Publicar no Power BI Service (requer conta)
   - Exportar como PDF para relatórios estáticos
   - Compartilhar o arquivo .pbix diretamente
"""
        
        # Salvando o arquivo de instruções
        with open(f'{output_path}power_bi_instructions.md', 'w') as f:
            f.write(instructions)
        
        print(f"Instruções criadas com sucesso em {output_path}power_bi_instructions.md")
        return True
    
    except Exception as e:
        print(f"Erro ao criar instruções: {e}")
        return False


if __name__ == "__main__":
    # Exemplo de uso do módulo
    print("Iniciando processo ETL...")
    
    # Extraindo dados
    raw_data = extract_data()
    
    if raw_data:
        # Transformando dados
        transformed_data = transform_data(raw_data)
        
        # Criando modelo dimensional
        dim_tables, fact_table = create_dimensional_model(transformed_data)
        
        # Criando tabelas agregadas
        agg_tables = create_aggregated_tables(fact_table, dim_tables)
        
        # Exportando para Power BI
        export_to_power_bi(dim_tables, fact_table, agg_tables)
        
        # Criando instruções para Power BI
        create_power_bi_instructions()
        
        print("Processo ETL concluído com sucesso!")
