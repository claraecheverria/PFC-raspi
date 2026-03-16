import AWSIoTPythonSDK
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import json
import time

def connect_iot_core(payload):

    # Identificador del cliente
    CLIENT_ID = "raspi-lab01"

    # ENDPOINT AWS IoT Core
    ENDPOINT = "a1ctxrgzb003b-ats.iot.us-east-1.amazonaws.com"

    # Topic configurado en Terraform
    TOPIC = "raspi5/sensors/data"

    # Paths a los certificados
    ROOT_CA = "/home/tesis/aws_certs/AmazonRootCA1.pem"
    CERT = "/home/tesis/aws_certs/certificate.pem.crt"
    PRIVATE = "/home/tesis/aws_certs/private.pem.key"

    client = AWSIoTMQTTClient(CLIENT_ID)
    client.configureEndpoint(ENDPOINT, 8883)
    client.configureCredentials(ROOT_CA, PRIVATE, CERT)

    client.configureOfflinePublishQueueing(-1)
    client.configureDrainingFrequency(2)
    client.configureConnectDisconnectTimeout(10)
    client.configureMQTTOperationTimeout(5)


    print("Conectando a AWS IoT...")
    client.connect()
    print("Conectado âœ”")


    client.publish(TOPIC, json.dumps(payload), 1)
    print("Enviado:", payload)
