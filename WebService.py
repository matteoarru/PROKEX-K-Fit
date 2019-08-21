from requests.auth import HTTPBasicAuth  # or HTTPDigestAuth, or OAuth1, etc.
from requests import Session
from zeep import Client
from zeep import xsd
from zeep.transports import Transport

#session = Session()
#session.headers.update({"Username": 'corvinnoStudio'})
#session.headers.update({"Password": '6ekNTES5MMcf3Em'})
#session.auth = HTTPBasicAuth('smartConsortium', 'UBFJQxRpbffjmZ2')
client = Client('http://studio.corvinno.hu:8080/StudioConnectionWS/services/StudioServices?wsdl')
 #           transport=Transport(session=session))

header = xsd.Element(
            'Security',
            xsd.ComplexType([
                xsd.Element(
                    'UsernameToken',
                    xsd.ComplexType([
                        xsd.Element('Username',xsd.String()),
                        xsd.Element('Password',xsd.String()),
                    ])
                
                ),
            ])
        )

header_value = header(UsernameToken={'Username':'corvinnoStudio','Password':'6ekNTES5MMcf3Em'})
#,UPSServiceAccessToken={'AccessLicenseNumber':'test_pwd'})

client.service.getLearningMaterial(_soapheaders=[header_value]),'TT-Sampling-28', 'EN'
                )



with client.options(raw_response=True):
    response = client.service.getLearningMaterial('TT-Sampling-28', 'EN')

    # response is now a regular requests.Response object
    assert response.status_code == 200
    assert response.content
print (client.service.getLearningMaterial('TT-Sampling-28', 'EN'))
