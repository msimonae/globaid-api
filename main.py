import os
import re
import requests
import io
import google.generativeai as genai
from urllib.parse import urlparse, quote_plus
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
from typing import Optional, List
from PIL import Image

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# --- Configuração das Chaves de API ---
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not RAPIDAPI_KEY or not GEMINI_API_KEY:
    raise RuntimeError("🚨 ALERTA: Chaves de API não encontradas. Verifique seu arquivo .env ou as variáveis de ambiente do servidor.")

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    raise RuntimeError(f"🚨 ALERTA: Falha ao configurar a API do Gemini. Erro: {e}")

# Inicializa a aplicação FastAPI
app = FastAPI(
    title="Analisador e Otimizador de Produtos Amazon com IA",
    description="Uma API para extrair dados de produtos da Amazon, analisar inconsistências e gerar listings otimizados com IA.",
    version="2.0.0", # Versão incrementada com a nova funcionalidade
)

# --- Modelos Pydantic para validação ---
class AnalyzeRequest(BaseModel):
    amazon_url: HttpUrl

class AnalyzeResponse(BaseModel):
    report: str
    asin: str
    country: str
    product_image_url: Optional[str] = None
    product_photos: Optional[List[str]] = None

# <<< NOVO: Modelos Pydantic para a nova funcionalidade de otimização
class OptimizeRequest(BaseModel):
    amazon_url: HttpUrl

class OptimizeResponse(BaseModel):
    optimized_listing_report: str
    asin: str
    country: str

# --- Agentes de Extração de Dados ---

def extract_product_info_from_url(url: str) -> Optional[dict]:
    # (Sua função original - sem alterações)
    asin_match = re.search(r"/([dg]p|product)/([A-Z0-9]{10})", url, re.IGNORECASE)
    if not asin_match:
        asin_match = re.search(r"/([A-Z0-9]{10})(?:[/?]|$)", url, re.IGNORECASE)
        if not asin_match:
            print("ASIN não encontrado na URL:", url)
            return None
        asin = asin_match.group(1)
    else:
        asin = asin_match.group(2)
    hostname = urlparse(url).hostname
    if not hostname:
        print("Hostname não encontrado na URL:", url)
        return None
    country_map = {
        "amazon.com.br": "BR", "amazon.com": "US", "amazon.co.uk": "GB",
        "amazon.de": "DE", "amazon.ca": "CA", "amazon.fr": "FR",
        "amazon.es": "ES", "amazon.it": "IT", "amazon.co.jp": "JP",
        "amazon.in": "IN", "amazon.com.mx": "MX", "amazon.com.au": "AU",
    }
    country = next((country_map[key] for key in country_map if key in hostname), "US")
    return {"asin": asin, "country": country}

def get_product_details(asin: str, country: str) -> dict:
    # (Sua função original - sem alterações)
    api_url = "https://real-time-amazon-data.p.rapidapi.com/product-details"
    querystring = {"asin": asin, "country": country}
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"}
    try:
        response = requests.get(api_url, headers=headers, params=querystring, timeout=30)
        response.raise_for_status()
        data = response.json().get("data")
        if not data:
            raise HTTPException(status_code=404, detail="Produto não encontrado na API da Amazon ou dados indisponíveis.")
        return data
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Erro ao chamar a API da Amazon para detalhes: {e}")

# <<< NOVO: Agente para buscar reviews do produto
def get_product_reviews(asin: str, country: str) -> dict:
    """Busca os reviews de um produto."""
    api_url = "https://real-time-amazon-data.p.rapidapi.com/product-reviews"
    # page=1&page_size=10 para pegar os mais recentes. Sort por 'recent'
    querystring = {"asin": asin, "country": country, "sort_by": "recent", "page_size": "20"} 
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"}
    try:
        response = requests.get(api_url, headers=headers, params=querystring, timeout=30)
        response.raise_for_status()
        data = response.json().get("data", {})
        # Filtra e formata reviews positivos e negativos
        reviews = data.get("reviews", [])
        positive = [r['review_comment'] for r in reviews if r['review_star_rating'] >= 4]
        negative = [r['review_comment'] for r in reviews if r['review_star_rating'] <= 2]
        return {"positive_reviews": positive[:10], "negative_reviews": negative[:10]} # Limita a 10 de cada
    except requests.exceptions.RequestException:
        # Se falhar, retorna uma lista vazia para não quebrar o processo
        return {"positive_reviews": [], "negative_reviews": []}

# <<< NOVO: Agente para buscar concorrentes
def get_competitors(keyword: str, country: str, original_asin: str) -> list:
    """Busca produtos na Amazon por palavra-chave para encontrar concorrentes."""
    api_url = "https://real-time-amazon-data.p.rapidapi.com/search"
    querystring = {"query": quote_plus(keyword), "country": country, "page_size":"10"}
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"}
    try:
        response = requests.get(api_url, headers=headers, params=querystring, timeout=30)
        response.raise_for_status()
        data = response.json().get("data", {})
        products = data.get("products", [])
        
        competitors = []
        for p in products:
            # Filtra o próprio produto e resultados patrocinados
            if p.get('asin') != original_asin and not p.get('is_sponsored', False):
                competitor_info = {
                    "title": p.get('product_title'),
                    "price": p.get('product_price'),
                    "rating": p.get('product_star_rating'),
                    "reviews_count": p.get('product_num_ratings')
                }
                competitors.append(competitor_info)
            if len(competitors) >= 5:
                break # Pega os 5 primeiros concorrentes orgânicos
        return competitors
    except requests.exceptions.RequestException:
        return []

# --- Agente 3: Analisador de Inconsistências (Função Original) ---
def analyze_product_with_gemini(product_data: dict) -> str:
    # (Sua função original - sem alterações)
    if not product_data:
        return "Não foi possível obter os dados do produto para análise."
    product_dimensions_text = "N/A"
    if info_table := product_data.get("product_information"):
        for key, value in info_table.items():
            if "dimens" in key.lower():
                product_dimensions_text = value
                break
    title = product_data.get("product_title", "N/A")
    features = "\n- ".join(product_data.get("about_product", []))
    image_urls = product_data.get("product_photos", [])
    if not image_urls:
        return "Produto sem imagens para análise."
    prompt_parts = [
        "Você é um analista de controle de qualidade de e-commerce extremamente meticuloso e detalhista.",
        "Sua principal tarefa é encontrar inconsistências factuais entre a descrição textual de um produto e suas imagens.",
        "Analise o texto e as imagens a seguir. Aponte QUALQUER discrepância, por menor que seja, entre o que está escrito e o que é mostrado. Preste atenção especial aos dados estruturados, como dimensões, peso, voltagem, cor, material e quantidade de itens.",
        "Se tudo estiver consistente, declare: 'Nenhuma inconsistência encontrada.'",
        "\n--- DADOS TEXTUAIS DO PRODUTO ---",
        f"**Título:** {title}",
        f"**Destaques (Sobre este item):**\n- {features}",
        f"**Dimensões do Produto (extraído da tabela de especificações):** {product_dimensions_text}",
        "\n--- IMAGENS PARA ANÁLISE VISUAL ---",
    ]
    image_count = 0
    for url in image_urls[:5]:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            prompt_parts.append(img)
            image_count += 1
        except requests.exceptions.RequestException as e:
            print(f"Aviso: Falha ao baixar a imagem {url}. Erro: {e}")
        except Exception as e:
            print(f"Aviso: Falha ao processar a imagem {url}. Erro: {e}")
    if image_count == 0:
        return "Nenhuma imagem pôde ser baixada para análise."
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao chamar a API do Gemini para análise: {e}")

# <<< NOVO: Agente 4: Otimizador de Listing com IA (Nova Função)
def optimize_listing_with_gemini(product_data: dict, reviews_data: dict, competitors_data: list, url_info: dict) -> str:
    """Usa a IA do Gemini com o prompt estratégico para otimizar o listing do produto."""
    
    # Mapeamento de país para idioma e mercado
    market_map = {
        "BR": ("Português (Brasil)", "Amazon BR"), "US": ("English (US)", "Amazon US"),
        "MX": ("Español (México)", "Amazon MX"), "ES": ("Español (España)", "Amazon ES"),
        # Adicione outros mercados conforme necessário
    }
    lang, market = market_map.get(url_info["country"], ("English (US)", f"Amazon {url_info['country']}"))

    # Extrai dados para o prompt
    current_title = product_data.get("product_title", "N/A")
    current_features = "\n- ".join(product_data.get("about_product", []))
    current_description = product_data.get("product_description", "N/A")

    # Constrói o prompt estratégico que criamos
    prompt_parts = [
        f"Você é um Consultor Sênior de E-commerce, mestre em SEO para o ecossistema Amazon (A9, Rufus) e especialista em otimização da taxa de conversão.",
        f"Sua missão é analisar profundamente o produto fornecido e gerar um listing completo e otimizado para o mercado alvo: {market}. O objetivo final é posicionar este produto para se tornar um best-seller, superando a concorrência.",
        f"A comunicação deve ser toda em {lang}.",
        "\n--- DADOS DO PRODUTO ATUAL ---",
        f"URL: {product_data.get('product_url')}",
        f"Título Atual: {current_title}",
        f"Feature Bullets Atuais: {current_features}",
        f"Descrição Atual: {current_description}",
        "\n--- INTELIGÊNCIA DE MERCADO E CONSUMIDOR ---",
        f"Insights de Reviews Positivos (use para destacar pontos fortes): {reviews_data.get('positive_reviews')}",
        f"Objeções de Reviews Negativos (use para criar contra-argumentos e FAQs): {reviews_data.get('negative_reviews')}",
        f"Análise dos Top 5 Concorrentes (use como benchmark para superar): {competitors_data}",
        "\n--- TAREFAS A SEREM EXECUTADAS ---",
        "Com base em todos os dados fornecidos, execute as seguintes tarefas e entregue o resultado em formato Markdown:",
        "1. **Título Otimizado (SEO):** Crie um título rico em palavras-chave, seguindo a fórmula: [Marca] + [Nome do Produto/Palavra-chave Principal] + [2-3 Benefícios/Características Chave] + [Tamanho/Quantidade].",
        "2. **Feature Bullets Otimizados (5 Pontos):** Reescreva 5 pontos. Cada um deve começar com um benefício claro em maiúsculas (ex: 'ENERGIA NATURAL:') seguido da característica. Use emojis discretos.",
        "3. **Descrição do Produto (Estrutura para A+ Content):** Crie uma descrição persuasiva com storytelling, detalhando os 3 principais benefícios e incluindo seções 'Como Usar' e 'Ideal Para'.",
        "4. **Análise Competitiva e Estratégia de Posicionamento:** Em uma tabela, compare meu produto com a média dos concorrentes (Preço, Features, Rating). Depois, defina o principal diferencial do meu produto.",
        "5. **Sugestões de Palavras-chave (Backend):** Liste 15-20 palavras-chave 'long-tail' para os 'Search Terms'.",
        "6. **FAQ Estratégico (Top 5 Perguntas e Respostas):** Crie 5 Q&As baseados nas dúvidas e objeções dos reviews.",
        "\n--- REGRAS INQUEBRÁVEIS ---",
        "- **NÃO INVENTE características.** Baseie-se estritamente nos dados fornecidos.",
        "- **NÃO USE clichês genéricos.** Seja específico e factual, focado em benefícios tangíveis.",
        "- **O conteúdo final deve ser único e superior ao dos concorrentes.**"
    ]
    
    try:
        # Usa um modelo mais robusto se necessário, mas flash é rápido e eficiente
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content("\n".join(prompt_parts))
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao chamar a API do Gemini para otimização: {e}")

# --- Endpoints da API ---

# Endpoint Original para Análise de Inconsistência
@app.post("/analyze", response_model=AnalyzeResponse)
def run_analysis_pipeline(request: AnalyzeRequest):
    amazon_url = str(request.amazon_url)
    url_info = extract_product_info_from_url(amazon_url)
    if not url_info:
        raise HTTPException(status_code=400, detail="URL inválida ou ASIN não encontrado.")
    product_data = get_product_details(url_info["asin"], url_info["country"])
    analysis_report = analyze_product_with_gemini(product_data)
    main_image_url = product_data.get("product_main_image_url") or (product_data.get("product_photos") or [None])[0]
    product_photos = product_data.get("product_photos") or []
    return AnalyzeResponse(
        report=analysis_report, asin=url_info["asin"], country=url_info["country"],
        product_image_url=main_image_url, product_photos=product_photos
    )

# <<< NOVO: Endpoint para Otimização Estratégica de Listing
@app.post("/optimize", response_model=OptimizeResponse)
def run_optimization_pipeline(request: OptimizeRequest):
    """
    Recebe uma URL da Amazon, executa a pipeline de otimização e retorna o listing completo.
    """
    amazon_url = str(request.amazon_url)

    # Agente 1: Extrair ASIN e País
    url_info = extract_product_info_from_url(amazon_url)
    if not url_info:
        raise HTTPException(status_code=400, detail="URL inválida ou ASIN não encontrado.")
    
    asin = url_info["asin"]
    country = url_info["country"]

    # Agente 2: Coletar todos os dados necessários
    product_data = get_product_details(asin, country)
    reviews_data = get_product_reviews(asin, country)
    
    # Usa o título do produto como palavra-chave inicial para encontrar concorrentes
    keyword_for_search = product_data.get("product_title", asin)
    competitors_data = get_competitors(keyword_for_search, country, asin)
    
    # Agente 4 (Novo): Gerar o listing otimizado
    optimization_report = optimize_listing_with_gemini(product_data, reviews_data, competitors_data, url_info)

    return OptimizeResponse(
        optimized_listing_report=optimization_report,
        asin=asin,
        country=country
    )
