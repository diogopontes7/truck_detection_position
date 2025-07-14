from datetime import datetime
import json
import os

import requests
from .inference import check_truck_position


def send_alignment_results_to_api(webhook_url='http://localhost/webhook:5000', truck_status=None):
    # Enviar resultados para a nossa API
    try:
        # Verificar o estado atraves da função que verifica se o camiao esta alinhado e se a front esta a frente da side
        if truck_status is None:
            truck_status = check_truck_position()
        
        # Determine overall status
        overall_aligned = truck_status['alinhado_vertical'] and truck_status['is_behind_side']
        status = "OK" if overall_aligned else "NOT_OK"
        
        # Create message based on alignment status
        if overall_aligned:
            message = "Truck is properly aligned and positioned correctly."
        else:
            issues = []
            if not truck_status['alinhado_vertical']:
                issues.append("vertical alignment issue")
            if not truck_status['is_behind_side']:
                issues.append("positioning behind side panel issue")
            message = f"Truck needs adjustment: {', '.join(issues)}."
        
        # objeto python para ser enviado como JSON
        payload = {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "message": message,
            "details": {
                "vertically_aligned": truck_status['alinhado_vertical'],
                "behind_side_panel": truck_status['is_behind_side'],
                "distance_to_side": round(truck_status['distance'], 2),
                "x_coordinates": truck_status['x_coords']
            },
            "image_filename": "C:/Users/Diogo/OneDrive - Universidade do Minho/Uni/Summer_Intern_Project/truck_detection_position/app/examples/0013247.jpg" 
        }
        
        # Send POST request
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Truck-Alignment-System/1.0'
        }
        
        response = requests.post(
            webhook_url, 
            json_data=json.dumps(payload), # torna o objeto python em JSON
            headers=headers,
        )
        
        # Check if request was successful
        if response.status_code in [200, 201, 202]:
            print(f"Successfully sent results to {webhook_url}")
            print(f"  Status: {status}")
            print(f"  Response: {response.status_code}")
            return True
        else:
            print(f"Failed to send results. HTTP {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Network error sending results: {e}")
        return False
    except Exception as e:
        print(f"Error preparing/sending results: {e}")
        return False

def check_and_notify(webhook_url=None, send_notification=True):
    """
    Complete truck alignment check with optional API notification
    
    Args:
        webhook_url (str): URL to send results to (if None, uses environment variable)
        send_notification (bool): Whether to send notification to API
    
    Returns:
        dict: Complete truck alignment results
    """
    print("🚛 Starting truck alignment check...")
    
    # Run the alignment check
    truck_status = check_truck_position()
    
    # Determine overall status
    overall_aligned = truck_status['alinhado_vertical'] and truck_status['is_behind_side']
    
    # Print results
    print(f"\n📊 Alignment Results:")
    print(f"  • Vertical Alignment: {'✓ PASS' if truck_status['alinhado_vertical'] else '✗ FAIL'}")
    print(f"  • Behind Side Panel: {'✓ PASS' if truck_status['is_behind_side'] else '✗ FAIL'}")
    print(f"  • Distance to Side: {truck_status['distance']:.2f}px")
    print(f"  • Overall Status: {'🟢 OK' if overall_aligned else '🔴 NOT OK'}")
    
    # Send notification if requested
    if send_notification:
        # Get webhook URL from environment if not provided
        if webhook_url is None:
            webhook_url = os.getenv('WEBHOOK_URL')
        
        if webhook_url:
            print(f"\n Sending notification to API...")
            success = send_alignment_results_to_api(webhook_url, truck_status)
        else:
            print("\nNo webhook URL provided. Set WEBHOOK_URL environment variable or pass webhook_url parameter.")
    
    # Add overall status to truck_status
    truck_status['overall_status'] = 'OK' if overall_aligned else 'NOT_OK'
    truck_status['overall_aligned'] = overall_aligned
    
    return truck_status

if __name__ == "__main__":
    # Run the check and notify
    results = check_and_notify(webhook_url='http://localhost:5000/webhook', send_notification=True)
    
    # Print final results
    print("\n📋 Final Results:")
    #print(results, indent=4, ensure_ascii=False)