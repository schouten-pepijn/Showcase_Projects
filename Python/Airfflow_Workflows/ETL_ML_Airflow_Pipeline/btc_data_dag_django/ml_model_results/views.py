from django.shortcuts import render
from django.http import JsonResponse
import json
from django.views.decorators.csrf import csrf_exempt

stored_data = []

# Create your views here.
@csrf_exempt
def model_results(request):
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            stored_data.append(data)
            return JsonResponse({'received': data})
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON data."}, status=400)
    elif request.method == 'GET':
        return JsonResponse({'data': stored_data})
    return JsonResponse({"error": "Invalid request method."}, status=405)