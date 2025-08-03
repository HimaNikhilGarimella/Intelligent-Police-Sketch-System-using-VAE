# Intelligent-Police-Sketch-System-using-VAE

# Abstract 
Sketch Generation in law enforcement involves of generating a visual representation of the suspect based on witness description as they aid to speed up investigation. The key objectives of the project are to develop an image that can be matched with the criminal database to speed up the investigation procedure, it also seeks to improve efficiency and accuracy of sketching by using interactive slider features. 	The methodology follows three phases, Train a Variational Auto Encoder (VAE) model on a large facial dataset to generate realistic sketches, Utilize Facenet-512 to match the generated sketch with the dataset and filter the top n (depends on number of matches) matches using Decision tree algorithm, Implement a User Interface with sliders to change facial features to get much accurate outputs. The novelty lies in combining VAE driven sketch generator with Facenet based matching for precise identification of suspect from rough sketch. Additionally, the interactive slider interface allows for more accurate results and those outputs can be added back to dataset for more training data. 

# Objectives of the project
1. To generate suspect sketches based on witness descriptions using a Variational Autoencoder (VAE) model.
2. To match generated sketches with the database using Facenet-512 with which law enforcements can identify the suspects quickly. 
3. To provide slider based interface so that features can be adjusted and fine tuned for better results. 
4. To improve the model by adding unidentified images to the database. 
5. To reduce manual sketching and improving overall efficiency.  

# Architecture Diagram

<img width="1500" height="689" alt="image" src="https://github.com/user-attachments/assets/d987ed8d-7ffe-4a5c-b028-819412efb676" />

# Dataset Details 
Scraped from the Illinois DOC.

https://www.idoc.state.il.us/subsections/search/inms_print.asp?idoc=
https://www.idoc.state.il.us/subsections/search/pub_showfront.asp?idoc=
https://www.idoc.state.il.us/subsections/search/pub_showside.asp?idoc=

Statistics for hair:
  43305 Black, 
  17371 Brown, 
   2887 Blonde or Strawberry, 
   2539 Gray or Partially Gray, 
    740 Red or Auburn, 
    624 Bald, 
    209 Salt and Pepper, 
     70 White, 
      7 Sandy,  

Statistics for sex:
  63409 Male, 
   4740 Female

Statistics for race:
  37991 Black, 
  20992 White, 
   8637 Hispanic, 
    235 Asian, 
    104 Amer Indian, 
     92 Bi-Racial

Statistics for eyes:
  51714 Brown, 
   7808 Blue, 
   4259 Hazel, 
   2469 Green, 
   1382 Black, 
     87 Gray.


