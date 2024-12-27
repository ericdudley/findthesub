from django.shortcuts import render
from django.http import JsonResponse
from dotenv import load_dotenv

from server.utils import query_closest_subs

def index(request):
    query = request.GET.get('q')
    if not query:
        return JsonResponse({'error': 'Missing required "q" query param. Example: /api?q=\'Test query\''}, status=400)
    
    try:
        # Use the new Zilliz-based approach
        zilliz_results = query_closest_subs(query, limit=1024, top_n=15)
        return JsonResponse({'zillizResults': zilliz_results["results"].to_dict(orient='records'), 'posts': zilliz_results["posts"].head(20).to_dict(orient="records")})
    except ValueError as e:
        return JsonResponse({'error': str(e)}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)