# main.py
import os
import re
import io
import requests
import google.generativeai as genai
from urllib.parse import urlparse, quote_plus
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
from typing import Optional, List
from PIL import Image

# Carrega as variáveis de ambiente
load_dotenv()

# --- Configuração das Chaves de API ---
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not RAPIDAPI_KEY or not GEMINI_API_KEY:
    raise RuntimeError("🚨 ALERTA: Chaves de API não encontradas.")

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    raise RuntimeError(f"🚨 ALERTA: Falha ao configurar a API do Gemini. Erro: {e}")

# Inicializa a aplicação FastAPI
app = FastAPI(
    title="Analisador e Otimizador de Produtos Amazon com IA",
    description="Uma API para extrair dados, analisar inconsistências e otimizar listings.",
    version="2.5.0", # Versão com prompt de análise avançado
)

# --- Modelos Pydantic ---
class AnalyzeRequest(BaseModel):
    amazon_url: HttpUrl

class AnalyzeResponse(BaseModel):
    report: str
    asin: str
    country: str
    product_title: Optional[str] = None
    product_image_url: Optional[str] = None
    product_photos: Optional[List[str]] = []

# ... (outros modelos Pydantic não precisam de alteração) ...

# --- Agentes de Extração de Dados ---
# (As funções extract_product_info_from_url, get_product_details, get_product_reviews, 
# e get_competitors não precisam de alterações)
def extract_product_info_from_url(url: str) -> Optional[dict]:
    asin_match = re.search(r"/([dg]p|product)/([A-Z0-9]{10})", url, re.IGNORECASE)
    if not asin_match:
        asin_match = re.search(r"/([A-Z0-9]{10})(?:[/?]|$)", url, re.IGNORECASE)
        if not asin_match: return None
        asin = asin_match.group(1)
    else:
        asin = asin_match.group(2)
    hostname = urlparse(url).hostname
    if not hostname: return None
    country_map = {"amazon.com.br": "BR", "amazon.com": "US", "amazon.co.uk": "GB", "amazon.de": "DE", "amazon.ca": "CA", "amazon.fr": "FR", "amazon.es": "ES", "amazon.it": "IT", "amazon.co.jp": "JP", "amazon.in": "IN", "amazon.com.mx": "MX", "amazon.com.au": "AU"}
    country = next((country_map[key] for key in country_map if key in hostname), "US")
    return {"asin": asin, "country": country}

def get_product_details(asin: str, country: str) -> dict:
    api_url = "https://real-time-amazon-data.p.rapidapi.com/product-details"
    querystring = {"asin": asin, "country": country}
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"}
    try:
        response = requests.get(api_url, headers=headers, params=querystring, timeout=30)
        response.raise_for_status()
        data = response.json().get("data")
        if not data: raise HTTPException(status_code=404, detail="Produto não encontrado na API da Amazon.")
        return data
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Erro ao chamar a API da Amazon para detalhes: {e}")


# --- Agente 3: Analisador de Inconsistências (PROMPT TOTALMENTE REFEITO) ---
def analyze_product_with_gemini(product_data: dict, country: str) -> str:
    if not product_data:
        return "Não foi possível obter os dados do produto para análise."

    title = product_data.get("product_title", "N/A")
    features = "\n- ".join(product_data.get("about_product", []))
    image_urls = product_data.get("product_photos", [])
    
    if not image_urls:
        return "Produto sem imagens para análise."

    # <<< CORREÇÃO: Prompt muito mais específico e detalhado
    prompt_parts = [
        "Você é um especialista em análise de conformidade de listings de e-commerce, focado em detalhes visuais e textuais.",
        "Sua tarefa é realizar uma análise multimodal detalhada, comparando os DADOS TEXTUAIS de um produto com o conteúdo visual de suas IMAGENS NUMERADAS.",
        "Seu objetivo é identificar 3 tipos de inconsistências:",
        "1. **CONTRADIÇÃO DIRETA:** O texto afirma algo (ex: cor, tamanho, quantidade) que a imagem claramente contradiz.",
        "2. **OMISSÃO NO TEXTO:** Uma imagem mostra uma informação crucial (como uma dimensão, selo de certificação, material, acessório incluído) que NÃO é mencionada nos dados textuais. Este é o tipo mais importante de inconsistência.",
        "3. **FALTA DE EVIDÊNCIA VISUAL:** O texto descreve um benefício ou característica importante que nenhuma das imagens fornecidas consegue comprovar visualmente.",
        "Para cada inconsistência encontrada, você DEVE citar o número da imagem específica (ex: 'Na Imagem 2, ...').",
        "Analise TODAS as imagens em busca de texto, logos, selos e detalhes. Dê prioridade máxima a dados estruturados como dimensões e especificações técnicas visíveis NAS IMAGENS e compare-os com o texto.",
        "Se, após uma análise minuciosa, tudo estiver consistente, declare: 'Nenhuma inconsistência factual encontrada.'",
        "\n--- DADOS TEXTUAIS DO PRODUTO ---",
        f"**Título:** {title}",
        f"**Destaques (Sobre este item):**\n- {features}",
        "\n--- IMAGENS PARA ANÁLISE VISUAL (numeradas sequencialmente a partir de 1) ---",
    ]
    
    image_count = 0
    for url in image_urls[:5]:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            # Adiciona um rótulo de texto antes de cada imagem para a IA entender a numeração
            prompt_parts.append(f"--- Imagem {image_count + 1} ---")
            prompt_parts.append(img)
            image_count += 1
        except Exception as e:
            print(f"Aviso: Falha ao processar a imagem {url}. Erro: {e}")
            
    if image_count == 0:
        return "Nenhuma imagem pôde ser baixada para análise."
        
    try:
        # Usar um modelo mais robusto como o 1.5 Pro pode dar melhores resultados para tarefas complexas de OCR e análise
        model = genai.GenerativeModel("gemini-1.5-pro-latest") 
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao chamar a API do Gemini para análise: {e}")

# ... (Restante do código, incluindo get_product_reviews, get_competitors, optimize_listing_with_gemini e os endpoints /analyze e /optimize não precisam de alterações) ...
# O endpoint /analyze já chama a função corrigida, então nenhuma outra mudança é necessária.
@app.post("/analyze", response_model=AnalyzeResponse)
def run_analysis_pipeline(request: AnalyzeRequest):
    url_info = extract_product_info_from_url(str(request.amazon_url))
    if not url_info:
        raise HTTPException(status_code=400, detail="URL inválida ou ASIN não encontrado.")

    product_data = get_product_details(url_info["asin"], url_info["country"])
    analysis_report = analyze_product_with_gemini(product_data, url_info["country"])

    return AnalyzeResponse(
        report=analysis_report,
        asin=url_info["asin"],
        country=url_info["country"],
        product_title=product_data.get("product_title"),
        product_image_url=product_data.get("product_main_image_url"),
        product_photos=product_data.get("product_photos", [])
    )
