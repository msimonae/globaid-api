# main_async.py (Versão Corrigida)
import os
import re
import httpx
import asyncio
from urllib.parse import urlparse, quote_plus, parse_qs
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
from typing import Optional, List
from openai import AsyncOpenAI

# Carrega as variáveis de ambiente
load_dotenv()

# --- Configuração das Chaves de API ---
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not RAPIDAPI_KEY or not OPENROUTER_API_KEY:
    raise RuntimeError("🚨 ALERTA: Chaves de API não encontradas. Verifique .env")

# --- <<< CORREÇÃO 1: IDs de Modelo Verificados e Funcionais ---
# Para tarefas rápidas e em lote (RECOMENDADO PARA /batch_analyze)
MODEL_ID_FAST = "google/gemini-2.5-flash"

# Para máxima qualidade de análise e raciocínio
MODEL_ID_PRO = "google/gemini-2.5-flash"

# Configura o cliente Async da API do OpenRouter
try:
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    print("✅ API Async do OpenRouter configurada com sucesso.")
except Exception as e:
    raise RuntimeError(f"🚨 ALERTA: Falha ao configurar a API do OpenRouter. Erro: {e}")

# Cria um cliente HTTP assíncrono global para reutilizar conexões
http_client = httpx.AsyncClient(timeout=45.0)

# Inicializa a aplicação FastAPI
app = FastAPI(
    title="Analisador e Otimizador de Produtos Amazon com IA",
    description="Uma API para extrair dados, analisar inconsistências e otimizar listings.",
    version="4.1.0", # Versão incrementada
)

# --- Modelos Pydantic (sem alterações) ---
class AnalyzeRequest(BaseModel):
    amazon_url: HttpUrl

class AnalyzeResponse(BaseModel):
    report: str
    asin: str
    country: str
    product_title: Optional[str] = None
    product_image_url: Optional[str] = None
    product_photos: Optional[List[str]] = []
    product_features: Optional[List[str]] = []

class BatchAnalyzeRequest(BaseModel):
    amazon_urls: List[HttpUrl]

class BatchAnalyzeResponse(BaseModel):
    results: List[AnalyzeResponse]

class OptimizeRequest(BaseModel):
    amazon_url: HttpUrl

class OptimizeResponse(BaseModel):
    optimized_listing_report: str
    asin: str
    country: str

# --- Mapeamento de Mercado (sem alterações) ---
MARKET_MAP = {
    "BR": ("Português (Brasil)", "Amazon BR"),
    "US": ("English (US)", "Amazon US"),
    "MX": ("Español (México)", "Amazon MX"),
    "ES": ("Español (España)", "Amazon ES"),
}

# --- Agentes de Extração de Dados ---
def extract_product_info_from_url(url: str) -> Optional[dict]:
    # (Função sem alterações)
    asin = None
    match = re.search(r"/([dg]p|product)/([A-Z0-9]{10})", url, re.IGNORECASE)
    if match:
        asin = match.group(2)
    else:
        match = re.search(r"/([A-Z0-9]{10})(?:[/?]|$)", url, re.IGNORECASE)
        if match:
            asin = match.group(1)
        else:
            try:
                parsed_url = urlparse(url)
                query_params = parse_qs(parsed_url.query)
                if 'asin' in query_params and re.match(r'^[A-Z0-9]{10}$', query_params['asin'][0]):
                    asin = query_params['asin'][0]
            except Exception:
                pass
    if not asin: return None
    hostname = urlparse(url).hostname
    if not hostname: return None
    country_map = {"amazon.com.br": "BR", "amazon.com": "US", "amazon.co.uk": "GB", "amazon.de": "DE", "amazon.ca": "CA", "amazon.fr": "FR", "amazon.es": "ES", "amazon.it": "IT", "amazon.co.jp": "JP", "amazon.in": "IN", "amazon.com.mx": "MX", "amazon.com.au": "AU"}
    country = next((country_map[key] for key in country_map if key in hostname), "US")
    return {"asin": asin, "country": country}

# --- <<< CORREÇÃO 2: Lógica Robusta para Extração de Imagens ---
async def get_product_details(asin: str, country: str) -> dict:
    api_url = "https://real-time-amazon-data.p.rapidapi.com/product-details"
    querystring = {"asin": asin, "country": country}
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"}
    try:
        response = await http_client.get(api_url, headers=headers, params=querystring)
        response.raise_for_status()
        data = response.json().get("data")
        if not data:
            raise HTTPException(status_code=404, detail="Produto não encontrado na API da Amazon.")

        # Lógica de extração de imagens resiliente
        image_keys_to_try = ['product_photos', 'images', 'product_images', 'image_urls']
        found_images = []
        for key in image_keys_to_try:
            potential_images = data.get(key)
            if isinstance(potential_images, list) and potential_images:
                found_images = potential_images
                break # Encontrou a lista de imagens, pode parar de procurar
        
        # Padroniza a chave de imagens para o resto da aplicação
        data['product_photos'] = found_images

        return data
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Erro ao chamar a API da Amazon para detalhes: {e}")

async def get_product_reviews(asin: str, country: str) -> dict:
    # (Função sem alterações)
    api_url = "https://real-time-amazon-data.p.rapidapi.com/product-reviews"
    querystring = {"asin": asin, "country": country, "sort_by": "recent", "page_size": "20"}
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"}
    try:
        response = await http_client.get(api_url, headers=headers, params=querystring)
        response.raise_for_status()
        reviews = response.json().get("data", {}).get("reviews", [])
        positive = [r['review_comment'] for r in reviews if r['review_star_rating'] >= 4]
        negative = [r['review_comment'] for r in reviews if r['review_star_rating'] <= 2]
        return {"positive_reviews": positive[:10], "negative_reviews": negative[:10]}
    except httpx.RequestError:
        return {"positive_reviews": [], "negative_reviews": []}

async def get_competitors(keyword: str, country: str, original_asin: str) -> list:
    # (Função sem alterações)
    api_url = "https://real-time-amazon-data.p.rapidapi.com/search"
    querystring = {"query": quote_plus(keyword), "country": country, "page_size":"10"}
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"}
    try:
        response = await http_client.get(api_url, headers=headers, params=querystring)
        response.raise_for_status()
        products = response.json().get("data", {}).get("products", [])
        competitors = []
        for p in products:
            if p.get('asin') != original_asin and not p.get('is_sponsored', False):
                competitors.append({"title": p.get('product_title'), "price": p.get('product_price'), "rating": p.get('product_star_rating'), "reviews_count": p.get('product_num_ratings')})
            if len(competitors) >= 5: break
        return competitors
    except httpx.RequestError:
        return []

# --- Agentes de IA ---
async def analyze_product_with_gemini(product_data: dict, country: str) -> str:
    # ... (lógica interna para montar o prompt é a mesma, sem alterações)
    product_dimensions_text = "N/A"
    info_table = product_data.get("product_information") or {}
    for key, value in info_table.items():
        if "dimens" in key.lower(): product_dimensions_text = value; break
    title = product_data.get("product_title", "N/A")
    description = product_data.get("product_description", "")
    features = product_data.get("about_product", []) or []
    features_text = "\n- ".join(features) if features else "N/A"
    full_text_content = f"{description}\n\nFeatures:\n- {features_text}".strip()
    image_urls = product_data.get("product_photos", []) or []
    if not image_urls:
        return (f"⚠️ Nenhuma imagem de produto foi encontrada na API.\n"
                f"--- DADOS TEXTUAIS ---\n"
                f"**Título:** {title}\n"
                f"**Conteúdo do anúncio:**\n{full_text_content}\n"
                f"**Dimensões (texto):** {product_dimensions_text}")
    prompt_parts = [
        "You are a meticulous e-commerce QA analyst specialized in numerical data validation and Amazon listings.",
        "Your task is to compare the TEXTUAL DATA of a product with its NUMBERED IMAGES to find factual contradictions, especially in dimensions, technical details, and numerical specifications.",
        "All your analysis and reasoning should be in English, but your final answer must be written entirely in **Portuguese**.",
    
        "Follow these steps:",
        "1. First, carefully examine EACH image and extract all visible numerical specifications (e.g., depth, width, height, weight, voltage, battery duration, etc.).",
        "2. Then, compare the extracted numbers from the images with the values found in the section 'TEXTUAL DATA'.",
        "3. If you find a numerical contradiction, clearly describe it, explicitly comparing the values from the text and the image.",
        "4. It is MANDATORY to mention the image number where the inconsistency was found (e.g., 'Na Imagem 2...').",
        "5. Analyze and compare Listing Data — textual content and product dimensions — and produce a clear and concise report listing ALL discrepancies found.",
        "6. Discrepancies may include:",
        "- Contradictory information (e.g., text says '10h battery' while image shows '8h battery').",
        "- Features mentioned in text but not visible or confirmed in images.",
        "- Important features visible in images but not mentioned in text.",
        "- Technical details (dimensions, weight, materials) inconsistent between text and images.",
        "- Any factual or visual inconsistency that could affect the customer’s purchase decision.",
        "- For each discrepancy, provide a short, objective justification explaining why it is considered an inconsistency.",
    
        "7. After factual analysis, evaluate whether the listing follows Amazon’s BEST PRACTICES:",
        "- Title: should include brand, product type, material, color/size, and not contain promotional terms.",
        "- Bullets: check for clarity, 5 bullet points, focus on benefits and differentiators.",
        "- Images: white background for main image, high resolution, lifestyle images, and different angles.",
        "- Description: should be clear, structured, focused on benefits and relevant technical info.",
        "- Keywords: ensure relevant search terms and SEO balance without excessive repetition.",
        "- Variations: verify if color/size options are correctly grouped under one listing.",
        "- Price and Stock: evaluate competitiveness and detect possible stock-out signs.",
        "- Reviews: ensure there are no mentions of ratings or reviews in the text.",
        "- A+ Content: identify presence of Enhanced Brand Content elements (if applicable).",
    
        "8. Finally, produce **two sections** in your final answer (in Portuguese):",
        "- **Inconsistências Fatuais entre Texto e Imagens** (if none, state 'Nenhuma inconsistência factual encontrada').",
        "- **Avaliação de Boas Práticas de Listing** (list strengths followed by improvement points).",
    
        "\n--- TEXTUAL DATA OF THE PRODUCT ---",
        f"**Título:** {title}",
        f"**Product Listing Text Content:**\n{full_text_content}",
        f"**Product Dimensions (text):** {product_dimensions_text}",
        "\n--- IMAGES FOR VISUAL ANALYSIS (numbered sequentially from 1) ---",
    ]
    
    for i, url in enumerate(image_urls[:5], start=1):
        prompt_parts.append(f"Image {i}: {url}")
    
    prompt_text = "\n".join(prompt_parts)

    try:
        response = await client.chat.completions.create(
            model=MODEL_ID_FAST, # Usa a constante corrigida
            messages=[{"role": "user", "content": prompt_text}],
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao chamar a API para análise: {e}")

async def optimize_listing_with_gemini(product_data: dict, reviews_data: dict, competitors_data: list, url_info: dict) -> str:
    # ... (lógica interna sem alterações)
    lang, market = MARKET_MAP.get(url_info["country"], ("English (US)", f"Amazon {url_info['country']}"))
    user_content = [ 
        f"Você é um Consultor Sênior de E-commerce, mestre em SEO para o ecossistema Amazon (A9, Rufus). Sua missão é otimizar um listing para maximizar vendas no mercado {market}.",
        f"A resposta DEVE ser inteiramente em {lang}.",
        f"--- DADOS DO PRODUTO ATUAL ---\nTítulo: {product_data.get('product_title', 'N/A')}\nFeatures: {product_data.get('about_product', [])}",
        f"--- INTELIGÊNCIA DE MERCADO ---\nReviews Positivos: {reviews_data.get('positive_reviews')}\nReviews Negativos: {reviews_data.get('negative_reviews')}\nConcorrentes: {competitors_data}",
        "\n--- INSTRUÇÕES E FORMATO DE SAÍDA OBRIGATÓRIO ---",
        "Gere sua resposta seguindo ESTRITAMENTE a estrutura Markdown abaixo, sem omitir nenhuma seção. Use os títulos exatamente como especificados.",
        "### 1. Título Otimizado (SEO)\n[Gere aqui o título otimizado]",
        "### 2. Feature Bullets Otimizados (5 Pontos)\n[Gere aqui os 5 feature bullets, um por linha]",
        "### 3. Descrição do Produto (Estrutura para A+ Content)\n[Gere aqui a descrição persuasiva]",
        "### 4. Análise Competitiva e Estratégia\n[Gere aqui a tabela comparativa e o parágrafo de estratégia]",
        "### 5. Sugestões de Palavras-chave (Backend)\n[Gere aqui a lista de 15-20 palavras-chave long-tail]",
        "### 6. FAQ Estratégico (Top 5 Perguntas e Respostas)\n[Gere aqui as 5 Q&As]",
        "\n--- REGRAS INQUEBRÁVEIS ---\n- Não invente características. Use apenas os dados fornecidos.\n- Não use clichês genéricos. Seja específico e factual.\n- O conteúdo final deve ser único e superior ao dos concorrentes."
    ]

    try:
        response = await client.chat.completions.create(
            model=MODEL_ID_PRO, # Usa a constante PRO para otimização de alta qualidade
            messages=[{"role": "user", "content": "\n".join(user_content)}]
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao chamar a API para otimização: {e}")

# --- Lógica de Processamento e Endpoints ---
async def process_single_url_async(url: str) -> AnalyzeResponse:
    # (Função sem alterações na sua lógica principal)
    url_info = extract_product_info_from_url(url)
    if not url_info:
        return AnalyzeResponse(report=f"Erro: URL inválida ou ASIN não encontrado.", asin="ERRO", country="N/A", product_title=f"Falha ao processar URL: {url}")
    try:
        product_data = await get_product_details(url_info["asin"], url_info["country"])
        analysis_report = await analyze_product_with_gemini(product_data, url_info["country"])
        return AnalyzeResponse(
            report=analysis_report,
            asin=url_info["asin"], country=url_info["country"],
            product_title=product_data.get("product_title"),
            product_image_url=product_data.get("product_main_image_url"),
            product_photos=product_data.get("product_photos", []),
            product_features=product_data.get("about_product", [])
        )
    except Exception as e:
        error_detail = getattr(e, 'detail', str(e))
        return AnalyzeResponse(
            report=f"Erro ao processar o ASIN {url_info.get('asin', 'N/A')}: {error_detail}",
            asin=url_info.get('asin', 'ERRO'), country=url_info.get('country', 'N/A'),
            product_title=f"Falha ao processar URL: {url}"
        )

@app.post("/analyze", response_model=AnalyzeResponse)
async def run_analysis_pipeline(request: AnalyzeRequest):
    # (Endpoint sem alterações)
    result = await process_single_url_async(str(request.amazon_url))
    if result.asin == "ERRO":
        raise HTTPException(status_code=400, detail=result.report)
    return result

@app.post("/batch_analyze", response_model=BatchAnalyzeResponse)
async def run_batch_analysis_pipeline(request: BatchAnalyzeRequest):
    # (Endpoint sem alterações)
    tasks = [process_single_url_async(str(url)) for url in request.amazon_urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            processed_results.append(AnalyzeResponse(
                report=f"Erro crítico durante o processamento concorrente: {str(result)}",
                asin="ERRO_FATAL", country="N/A"
            ))
        else:
            processed_results.append(result)
    return BatchAnalyzeResponse(results=processed_results)

@app.post("/optimize", response_model=OptimizeResponse)
async def run_optimization_pipeline(request: OptimizeRequest):
    # (Endpoint sem alterações)
    url_info = extract_product_info_from_url(str(request.amazon_url))
    if not url_info: raise HTTPException(status_code=400, detail="URL inválida ou ASIN não encontrado.")
    asin, country = url_info["asin"], url_info["country"]
    product_data, reviews_data = await asyncio.gather(
        get_product_details(asin, country),
        get_product_reviews(asin, country)
    )
    keyword = product_data.get("product_title", asin)
    competitors_data = await get_competitors(keyword, country, asin)
    optimization_report = await optimize_listing_with_gemini(product_data, reviews_data, competitors_data, url_info)
    return OptimizeResponse(
        optimized_listing_report=optimization_report,
        asin=asin, country=country
    )







