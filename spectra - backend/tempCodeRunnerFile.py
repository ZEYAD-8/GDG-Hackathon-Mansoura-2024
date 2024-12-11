
# import modelbit
# mb = modelbit.login()

# def prediction111(Age,Gender,Ethinicity,jaudice,Autism,Country,used_app,Relation,Result,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10):
#     data = pd.DataFrame([[Age,Gender,Ethinicity,jaudice,Autism,Country,used_app,Relation,Result,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10]],
#                        columns= ['Age','Gender','Ethnicity','Jaundice_born','Autism','Country','Used_app_before','Relation','result','A1_Score','A2_Score'	,'A3_Score','A4_Score',	'A5_Score','A6_Score','A7_Score','A8_Score','A9_Score',	'A10_Score'])
#     predict = pipe_predict.predict(data)
#     if check[0] == 0 :
#         final_predict = "No_Autism"
#     elif check[0] == 1 :
#          final_predict = "Autism"


#     return {
#         "prediction is":final_predict
#     }
# mb.deploy(prediction111)

# import requests
# import json

# # Define the input data correctly
# input_data = {
#     "data": [25, 'Male', 'Middle Eastern', 'Yes', 'No', 'Egypt', 'No', 'Parent', 9, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,]  
# }

# # Define the URL (replace <your-workspace-url> with the correct URL)
# url = "https://ahmedmostafa.us-east-2.aws.modelbit.com/v1/prediction111/latest"

# # Make the POST request
# response = requests.post(
#     url,
#     headers={"Content-Type": "application/json"},
#     json = input_data
# )
# response_json = response.json()

# # Check response
# print(json.dumps(response_json, indent=4))


