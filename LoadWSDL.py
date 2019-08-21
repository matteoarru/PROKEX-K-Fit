import pretend
from requests import Session
from requests.auth import HTTPBasicAuth  # or HTTPDigestAuth, or OAuth1, etc.
from zeep import Client
from zeep.transports import Transport
from zeep.cache import SqliteCache
transport = Transport(cache=SqliteCache())
session = Session()
session.auth = HTTPBasicAuth("gneusch", "gneusch")
client = Client('http://studio.corvinno.hu:8080/StudioConnectionWS/services/StudioServices?wsdl',
    transport=transport)

response = pretend.stub(
    status_code=200,
    headers={},
    content="""
    <!-- The response from the server -->
    """)
operation = client.service._binding._operations['getStatFill']
result = client.service._binding.process_reply(
        client, operation, response)
print(result)