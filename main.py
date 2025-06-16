import os
import re
import requests
import google.generativeai as genai
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env (para desenvolvimento local)
load_dotenv()

# --- Configuração das Chaves de API de forma segura ---
RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not RAPIDAPI_KEY or not GEMINI_API_KEY:
    print("🚨 ALERTA: Chaves de API não encontradas. Verifique seu arquivo .env ou as variáveis de ambiente do servidor.")

# Configura a API do Gemini globalmente
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Inicializa a aplicação FastAPI
app = FastAPI(
    title="Analisador de Produtos Amazon com IA",
    description="Uma API para extrair dados de produtos da Amazon e gerar relatórios de inconsistência com Gemini, alinhada com a lógica do Colab.",
    version="1.1.0"
)

# --- Modelos Pydantic para validação de entrada e saída ---
class AnalyzeRequest(BaseModel):
    amazon_url: HttpUrl

class AnalyzeResponse(BaseModel):
    report: str
    asin: str
    country: str

# --- Agente 1: Extrator de Informações da URL (ALINHADO COM COLAB) ---
def extract_product_info_from_url(url: str) -> dict:
    """Extrai o ASIN e o código do país de uma URL da Amazon."""
    asin_match = re.search(r'/(dp|gp/product)/([A-Z0-9]{10})', url)
    if not asin_match:
        return None
    asin = asin_match.group(2)
    hostname = urlparse(url).hostname
    
    # --- CORREÇÃO 1: Mapeamento de países expandido, igual ao do Colab ---
    country_map = {
        'amazon.com.br': 'BR', 'amazon.com': 'US', 'amazon.co.uk': 'GB', 
        'amazon.de': 'DE', 'amazon.ca': 'CA', 'amazon.fr': 'FR', 
        'amazon.es': 'ES', 'amazon.it': 'IT', 'amazon.co.jp': 'JP', 
        'amazon.in': 'IN', 'amazon.com.mx': 'MX', 'amazon.com.au': 'AU'
    }
    country = next((country_map[key] for key in country_map if key in hostname), 'US')
    return {"asin": asin, "country": country}

# --- Agente 2: Coletor de Detalhes do Produto (API da Amazon) ---
def get_product_details(asin: str, country: str) -> dict:
    """Busca detalhes de um produto na Amazon usando a API da RapidAPI."""
    api_url = "https://real-time-amazon-data.p.rapidapi.com/product-details"
    querystring = {"asin": asin, "country": country}
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"
    }
    try:
        response = requests.get(api_url, headers=headers, params=querystring, timeout=20)
        response.raise_for_status()
        data = response.json().get('data')
        if not data:
            raise HTTPException(status_code=404, detail="Produto não encontrado na API da Amazon ou dados indisponíveis.")
        return data
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Erro ao chamar a API da Amazon: {e}")

# --- Agente 3: Analisador com Gemini (IA) (ALINHADO COM COLAB) ---
def analyze_product_with_gemini(product_data: dict) -> str:
    """Usa a IA do Gemini para analisar inconsistências entre texto e imagens."""
    if not product_data:
        return "Não foi possível obter os dados do produto para análise."

    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    
    # Lógica de extração de dimensões idêntica à do Colab
    product_dimensions_text = 'N/A'
    if info_table := product_data.get('product_information'):
        for key, value in info_table.items():
            if 'dimens' in key.lower():
                product_dimensions_text = value
                break

    title = product_data.get('product_title', 'N/A')
    features = "\n- ".join(product_data.get('about_product', []))
    image_urls = product_data.get('product_photos', [])

    if not image_urls:
        return "Produto sem imagens para análise."

    # --- CORREÇÃO 2: Estrutura do prompt idêntica à do Colab ---
    prompt_parts = [
        "Você é um analista de controle de qualidade de e-commerce extremamente meticuloso.",
        "Sua principal tarefa é encontrar inconsistências factuais entre a descrição textual de um produto e suas imagens ou vídeos.",
        "Preste atenção especial aos dados estruturados, como dimensões, peso, voltagem e capacidade.",

        "\n--- DADOS TEXTUAIS DO PRODUTO ---",
        f"**Título:** {title}",
        f"**Destaques (Sobre este item):**\n- {features}",
        f"**Dimensões do Produto (extraído da tabela de especificações):** {product_dimensions_text}",

        "\n--- IMAGENS PARA ANÁLISE VISUAL ---",
    ]

    # --- CORREÇÃO 3: Remoção do limite de 5 imagens ---
    for url in image_urls:
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            mime_type = response.headers.get('Content-Type')
            if mime_type and 'image' in mime_type:
                prompt_parts.append({'inline_data': {'data': response.content, 'mime_type': mime_type}})
        except requests.exceptions.RequestException as e:
            print(f"Aviso: Não foi possível baixar a imagem {url}. Erro: {e}")
    
    # Restante do prompt, idêntico ao Colab
    prompt_parts.extend([
        "\n--- INSTRUÇÕES DETALHADAS PARA ANÁLISE ---",
        "1. **Análise de Dimensões (TAREFA PRINCIPAL):** Extraia as dimensões (Altura, Largura, Profundidade) dos DADOS TEXTUAIS. Em seguida, examine CUIDADOSAMENTE as imagens em busca de números e unidades (cm, kg, L) em gráficos ou textos sobrepostos. Compare os dois conjuntos de dados e relate QUALQUER discrepância, citando os valores exatos de cada fonte (Texto vs. Imagem).",
        "2. **Análise de Características:** Verifique se as características listadas nos Destaques são visualmente representadas ou compatíveis com as imagens.",
        "3. **Análise Geral:** Procure por outras inconsistências (cor, modelo, acessórios, voltagem).",

        "\n--- RELATÓRIO FINAL DE INCONSISTÊNCIAS ---",
        "Se encontrar discrepâncias, liste-as de forma clara e objetiva. Se estiver tudo correto, declare: 'Nenhuma inconsistência encontrada.'"
    ])

    try:
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao chamar a API do Gemini: {e}")


# --- Endpoint Principal da API ---
@app.post("/analyze", response_model=AnalyzeResponse)
def run_analysis_pipeline(request: AnalyzeRequest):
    """
    Recebe uma URL da Amazon, executa a pipeline de análise e retorna o relatório.
    """
    amazon_url = str(request.amazon_url)
    
    # Agente 1
    url_info = extract_product_info_from_url(amazon_url)
    if not url_info:
        raise HTTPException(status_code=400, detail="URL inválida ou ASIN não encontrado. Verifique a URL do produto.")
    
    # Agente 2
    product_data = get_product_details(url_info['asin'], url_info['country'])
    
    # Agente 3
    analysis_report = analyze_product_with_gemini(product_data)
    
    return AnalyzeResponse(
        report=analysis_report,
        asin=url_info['asin'],
        country=url_info['country']
    )