# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 14:49:24 2018

@author: matte
"""
import json
from requests.auth import HTTPBasicAuth  # or HTTPDigestAuth, or OAuth1, etc.
from requests import Session
from zeep import Client
from zeep import xsd
from zeep.transports import Transport
import pandas as pd

session = Session()
#session.auth = HTTPBasicAuth('corvinnoStudio', '6ekNTES5MMcf3Em')
session.headers.update({"Username": 'corvinnoStudio'})
session.headers.update({"Password": '6ekNTES5MMcf3Em'})
#session.auth = HTTPBasicAuth('smartConsortium', 'UBFJQxRpbffjmZ2')
client = Client('http://studio.corvinno.hu:8080/StudioConnectionWS/services/StudioServices?wsdl'
                , transport=Transport(session=session))
#client.service.getLearningMaterial('TT-Sampling-28', 'en')     
# createCGFromNodeList(nodeIdList: xsd:string, lang: xsd:string, depth: xsd:string, viewName: xsd:string, cGName: xsd:string) -> return: xsd:string
#client.service.createCGFromNodeList('TT-Sampling-28', 'en', '3', 'Test', 'MOL')
#{"selected":[{"word":"AF-Actual_Performance-13"},{"word":"AF-Adat-143"},{"word":"AF-Availability-44"},{"word":"AF-Contamination-27"},{"word":"AF-Delivery-36"},{"word":"AF-Guideline-140"},{"word":"AF-Kock�zat-144"},{"word":"AF-Performance-36"},{"word":"AF-pm_cost-126"},{"word":"AF-Purchase-36"},{"word":"AF-Quality_of_the_Sample-28"},{"word":"AF-Risk_Assessment-127"},{"word":"AF-Szervezet-141"},{"word":"TT-Acceptance_Procedure-1"},{"word":"TT-Ad_hoc_Sampling-28"},{"word":"TT-Asset-45"},{"word":"TT-Automatic_Tanker_Loading_Station-1"},{"word":"TT-Barge_Gauging-1"},{"word":"TT-Barge-1"},{"word":"TT-Chargeable_Loss-1"},{"word":"TT-Compliance_Objective-127"},{"word":"TT-Control_Measurement_Accuracy-1"},{"word":"TT-Cost_and_Resource_Analysis-127"},{"word":"TT-Cost_Reduction-36"},{"word":"TT-Customer_Order-1"},{"word":"TT-Dead_Stock-1"},{"word":"TT-Decision_Making_Process-17"},{"word":"TT-Discharging_Procedure-1"},{"word":"TT-Dispatcher-1"},{"word":"TT-Document-140"},{"word":"TT-Electronic_Dip_Stick-1"},{"word":"TT-Emptiness_Check-1"},{"word":"TT-Excise_Duty_Licence-1"},{"word":"TT-Excise_Duty_Regulation-1"},{"word":"TT-Filling_Station-1"},{"word":"TT-Finance_and_accounting-85"},{"word":"TT-Finance_Guard_Agency-1"},{"word":"TT-F�ldg�z-143"},{"word":"TT-Folyamat-143"},{"word":"TT-Forecasted_Daily_Sale-1"},{"word":"TT-Forecasting-126"},{"word":"TT-Free_Circulation_of_Goods-1"},{"word":"TT-Freight_Forwarding_Documentation-1"},{"word":"TT-Fuel_Density-1"},{"word":"TT-Gauge_System-1"},{"word":"TT-Governing_Law-126"},{"word":"TT-Handling_of_Contaminated_Disposal-92"},{"word":"TT-Hauling_Alongside-1"},{"word":"TT-Hullad�k_megeloz�s_�s_kezel�s-143"},{"word":"TT-Human_Resources-36"},{"word":"TT-Inventory_Level-1"},{"word":"TT-Inventory_Management-1"},{"word":"TT-Inventory_Replenishment_Systems-1"},{"word":"TT-Invoice-85"},{"word":"TT-ISO_Standards-28"},{"word":"TT-Law-69"},{"word":"TT-Loading_Gantry-1"},{"word":"TT-Loading_Procedure-1"},{"word":"TT-Logistic_Plan-1"},{"word":"TT-Logistics_Cost_and_Performance_Monitoring-1"},{"word":"TT-Logistics-1"},{"word":"TT-Loss_Regulation-1"},{"word":"TT-Metrological_Authority-1"},{"word":"TT-Metrological_Inspection-1"},{"word":"TT-Minimum_Delivery_Quantity-1"},{"word":"TT-Net_Quantity-1"},{"word":"TT-Non_Excise_Duty_Licensed_Trading-1"},{"word":"TT-Operation_and_Logistics-126"},{"word":"TT-Order_Management-1"},{"word":"TT-Order_Picking_and_Packing-1"},{"word":"TT-Performance_based_Evaluation_Measures-13"},{"word":"TT-Planned_Sampling-28"},{"word":"TT-Problem-17"},{"word":"TT-project_reporting-126"},{"word":"TT-Project_team-126"},{"word":"TT-Pump_Stock_Level-1"},{"word":"TT-Purchase_Order-1"},{"word":"TT-Rail_Transport-1"},{"word":"TT-Railway_Service-1"},{"word":"TT-Railway_Tank_Car-1"},{"word":"TT-Replenishment_Level-1"},{"word":"TT-Road_Freight_Routing_and_Scheduling-1"},{"word":"TT-Road_Freight_Transport-1"},{"word":"TT-Road_Weighing_Bridge-1"},{"word":"TT-Sample_Collection-28"},{"word":"TT-Sampling_Process-28"},{"word":"TT-Sampling_Technique-28"},{"word":"TT-Sampling-28"},{"word":"TT-Scheduling_in_SCM-125"},{"word":"TT-Selective_Sampling-28"},{"word":"TT-Shipping_Document-1"},{"word":"TT-Strategic_Performance_Indicator-17"},{"word":"TT-Supply_Source-1"},{"word":"TT-Takeover_Handover_Procedure-1"},{"word":"TT-Tank_Bottom_Loading-1"},{"word":"TT-Tank_Bottom_Residue-1"},{"word":"TT-Tank_Compartment-1"},{"word":"TT-Tank-1"},{"word":"TT-Tare_Weight-1"},{"word":"TT-Tax_Warehouse-1"},{"word":"TT-Transfer-126"},{"word":"TT-Transportation-1"},{"word":"TT-Travel_document-69"},{"word":"TT-Visual_Inspection-1"},{"word":"TT-Wagon-1"},{"word":"TT-Weighing_Bridge-1"}]}
with open("C://Users//matte//OneDrive//Thesis Matteo//prokexpy//concepts.txt") as f:
    selected = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
selected = [x.strip() for x in selected] 
#selected = [json.dumps({'word': x}) for x in selected]
jsonstring =json.dumps({'word': selected}) 
jsonstring ="{\"selected\":[{\"word\":\"AF-Actual_Performance-13\"},{\"word\":\"AF-Adat-143\"},{\"word\":\"AF-Availability-44\"},{\"word\":\"AF-Contamination-27\"},{\"word\":\"AF-Delivery-36\"},{\"word\":\"AF-Guideline-140\"},{\"word\":\"AF-Kock�zat-144\"},{\"word\":\"AF-Performance-36\"},{\"word\":\"AF-pm_cost-126\"},{\"word\":\"AF-Purchase-36\"},{\"word\":\"AF-Quality_of_the_Sample-28\"},{\"word\":\"AF-Risk_Assessment-127\"},{\"word\":\"AF-Szervezet-141\"},{\"word\":\"TT-Acceptance_Procedure-1\"},{\"word\":\"TT-Ad_hoc_Sampling-28\"},{\"word\":\"TT-Asset-45\"},{\"word\":\"TT-Automatic_Tanker_Loading_Station-1\"},{\"word\":\"TT-Barge_Gauging-1\"},{\"word\":\"TT-Barge-1\"},{\"word\":\"TT-Chargeable_Loss-1\"},{\"word\":\"TT-Compliance_Objective-127\"},{\"word\":\"TT-Control_Measurement_Accuracy-1\"},{\"word\":\"TT-Cost_and_Resource_Analysis-127\"},{\"word\":\"TT-Cost_Reduction-36\"},{\"word\":\"TT-Customer_Order-1\"},{\"word\":\"TT-Dead_Stock-1\"},{\"word\":\"TT-Decision_Making_Process-17\"},{\"word\":\"TT-Discharging_Procedure-1\"},{\"word\":\"TT-Dispatcher-1\"},{\"word\":\"TT-Document-140\"},{\"word\":\"TT-Electronic_Dip_Stick-1\"},{\"word\":\"TT-Emptiness_Check-1\"},{\"word\":\"TT-Excise_Duty_Licence-1\"},{\"word\":\"TT-Excise_Duty_Regulation-1\"},{\"word\":\"TT-Filling_Station-1\"},{\"word\":\"TT-Finance_and_accounting-85\"},{\"word\":\"TT-Finance_Guard_Agency-1\"},{\"word\":\"TT-F�ldg�z-143\"},{\"word\":\"TT-Folyamat-143\"},{\"word\":\"TT-Forecasted_Daily_Sale-1\"},{\"word\":\"TT-Forecasting-126\"},{\"word\":\"TT-Free_Circulation_of_Goods-1\"},{\"word\":\"TT-Freight_Forwarding_Documentation-1\"},{\"word\":\"TT-Fuel_Density-1\"},{\"word\":\"TT-Gauge_System-1\"},{\"word\":\"TT-Governing_Law-126\"},{\"word\":\"TT-Handling_of_Contaminated_Disposal-92\"},{\"word\":\"TT-Hauling_Alongside-1\"},{\"word\":\"TT-Hullad�k_megeloz�s_�s_kezel�s-143\"},{\"word\":\"TT-Human_Resources-36\"},{\"word\":\"TT-Inventory_Level-1\"},{\"word\":\"TT-Inventory_Management-1\"},{\"word\":\"TT-Inventory_Replenishment_Systems-1\"},{\"word\":\"TT-Invoice-85\"},{\"word\":\"TT-ISO_Standards-28\"},{\"word\":\"TT-Law-69\"},{\"word\":\"TT-Loading_Gantry-1\"},{\"word\":\"TT-Loading_Procedure-1\"},{\"word\":\"TT-Logistic_Plan-1\"},{\"word\":\"TT-Logistics_Cost_and_Performance_Monitoring-1\"},{\"word\":\"TT-Logistics-1\"},{\"word\":\"TT-Loss_Regulation-1\"},{\"word\":\"TT-Metrological_Authority-1\"},{\"word\":\"TT-Metrological_Inspection-1\"},{\"word\":\"TT-Minimum_Delivery_Quantity-1\"},{\"word\":\"TT-Net_Quantity-1\"},{\"word\":\"TT-Non_Excise_Duty_Licensed_Trading-1\"},{\"word\":\"TT-Operation_and_Logistics-126\"},{\"word\":\"TT-Order_Management-1\"},{\"word\":\"TT-Order_Picking_and_Packing-1\"},{\"word\":\"TT-Performance_based_Evaluation_Measures-13\"},{\"word\":\"TT-Planned_Sampling-28\"},{\"word\":\"TT-Problem-17\"},{\"word\":\"TT-project_reporting-126\"},{\"word\":\"TT-Project_team-126\"},{\"word\":\"TT-Pump_Stock_Level-1\"},{\"word\":\"TT-Purchase_Order-1\"},{\"word\":\"TT-Rail_Transport-1\"},{\"word\":\"TT-Railway_Service-1\"},{\"word\":\"TT-Railway_Tank_Car-1\"},{\"word\":\"TT-Replenishment_Level-1\"},{\"word\":\"TT-Road_Freight_Routing_and_Scheduling-1\"},{\"word\":\"TT-Road_Freight_Transport-1\"},{\"word\":\"TT-Road_Weighing_Bridge-1\"},{\"word\":\"TT-Sample_Collection-28\"},{\"word\":\"TT-Sampling_Process-28\"},{\"word\":\"TT-Sampling_Technique-28\"},{\"word\":\"TT-Sampling-28\"},{\"word\":\"TT-Scheduling_in_SCM-125\"},{\"word\":\"TT-Selective_Sampling-28\"},{\"word\":\"TT-Shipping_Document-1\"},{\"word\":\"TT-Strategic_Performance_Indicator-17\"},{\"word\":\"TT-Supply_Source-1\"},{\"word\":\"TT-Takeover_Handover_Procedure-1\"},{\"word\":\"TT-Tank_Bottom_Loading-1\"},{\"word\":\"TT-Tank_Bottom_Residue-1\"},{\"word\":\"TT-Tank_Compartment-1\"},{\"word\":\"TT-Tank-1\"},{\"word\":\"TT-Tare_Weight-1\"},{\"word\":\"TT-Tax_Warehouse-1\"},{\"word\":\"TT-Transfer-126\"},{\"word\":\"TT-Transportation-1\"},{\"word\":\"TT-Travel_document-69\"},{\"word\":\"TT-Visual_Inspection-1\"},{\"word\":\"TT-Wagon-1\"},{\"word\":\"TT-Weighing_Bridge-1\"}]}"
jsonstring ="{\"selected\":[{\"word\":\"AF-Purchase-36\"},{\"word\":\"AF-Outcome-127\"},{\"word\":\"TT-Product-78\"}]}"
jsonstring ="{\"selected\":[{\"word\":\"RR-Acwerwewormance-13\"},{\"word\":\"AF-Adat-143\"},{\"word\":\"AF-Availability-44\"}]}"



myvar=client.service.createCGFromNodeList(jsonstring , 'en', '1', 'MOL_LOG5', 'MOL_LOG5')
print(myvar)
client.service.getLearningMaterial('AF-Contamination-27', 'en')     


for y in selected:
        if client.service.getLearningMaterial(y, 'en')=='Empty content':
                print(y)
