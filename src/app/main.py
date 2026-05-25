from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import predict

#Initialise FastAPI
app = FastAPI(title='Customer Churn Prediction API',
              description='ML API for predicting customer churn in telcom industry',
              version='1.0.0')

#Root endpoint
#Critical: required for AWS Application Load Balancer health checks
@app.get('/')
def root():
    return {'message': 'FastAPI is running'}

#pydantic model for automatic validation and API documentation
class CustomerData(BaseModel):

    #Demographics
    gender: str                # Male or Female
    SeniorCitizen: int              
    Partner: str               #Yes or No - has partner
    Dependents: str            #Yes or No - has dependents

    #Phone services 
    PhoneService: str           #Yes or No
    MultipleLines: str          #Yes, No or No phone service

    #Internet service
    InternetService: str         #DSL, Fiber optics or No
    OnlineSecurity: str          #Yes, No or No internet service
    OnlineBackup: str            #Yes, No or No internet service
    DeviceProtection: str        #Yes, No or No internet service
    TechSupport: str             #Yes, No or No internet service
    StreamingTV: str             #Yes, No or No internet service
    StreamingMovies: str         #Yes, No or No internet service

    #Account information 
    Contract: str                #Month-to-month, One year, Two year
    PaperlessBilling: str        #Yes or No
    PaymentMethod: str           #Electronic check, Mailed check, etc

    #Numeric features
    tenure: int                  #number of months with company
    MonthlyCharges: float        #monthly charges in dollars
    TotalCharges: float          #total charges to date

#main prediction API endpoint
@app.post('/predict')
def get_prediction(data: CustomerData):
    try:

        #convert pydantic model to dict and call inference pipeline 
        result = predict(data.dict())

        return result
    
    except Exception as e:
        #return error details for debagging (consider logging in production)
        return {'error': str(e)}
