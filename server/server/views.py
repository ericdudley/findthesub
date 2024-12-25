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
        zilliz_results = query_closest_subs(query, limit=1000, top_n=5)
        return JsonResponse({'zillizResults': zilliz_results.to_dict(orient='records')})
    except ValueError as e:
        return JsonResponse({'error': str(e)}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)