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

paste <(cat ids.txt | sed 's/^/http:\/\/www.idoc.state.il.us\/subsections\/search\/pub_showside.asp\?idoc\=/g') <(cat ids.txt| sed 's/^/  out=/g' | sed 's/$/.jpg/g') -d '\n' > showside.txt 
paste <(cat ids.txt | sed 's/^/http:\/\/www.idoc.state.il.us\/subsections\/search\/pub_showfront.asp\?idoc\=/g') <(cat ids.txt| sed 's/^/  out=/g' | sed 's/$/.jpg/g') -d '\n' > showfront.txt      
paste <(cat ids.txt | sed 's/^/http:\/\/www.idoc.state.il.us\/subsections\/search\/inms_print.asp\?idoc\=/g') <(cat ids.txt| sed 's/^/  out=/g' | sed 's/$/.html/g') -d '\n' > inmates_print.txt 

aria2c -i ../inmates_print.txt -j4 -x4 -l ../log-$(pwd|rev|cut -d/ -f 1|rev)-$(date +%s).txt

Then use htmltocsv.py to get the csv. Note that the script is very poorly written and may have errors. It also doesn't do anything with the warrant-related info, although there are some commented-out lines which may be relevant.
Also note that it assumes all the HTML files are located in the inmates directory., and overwrites any csv files in csv if there are any.

front.7z contains mugshots from the front
side.7z contains mugshots from the side
inmates.7z contains all the html files
csv contains the html files converted to CSV

The reason for packaging the images is that many torrent clients would otherwise crash if attempting to load the torrent.

All CSV files contain headers describing the nature of the columns. For person.csv, the id is unique. For marks.csv and sentencing.csv, it is not.
Note that the CSV files use semicolons as delimiters and also end with a trailing semicolon. If this is unsuitable, edit the arr2csvR function in htmltocsv.py.

There are 68149 inmates in total, although some (a few hundred) are marked as "Unknown"/"N/A"/"" in one or more fields.

The "height" column has been processed to contain the height in inches, rather than the height in feet and inches expressed as "X ft YY in."
Some inmates were marked "Not Available", this has been replaced with "N/A".
Likewise, the "weight" column has been altered "XXX lbs." -> "XXX". Again, some are marked "N/A".

The "date of birth" column has some inmates marked as "Not Available" and others as "". There doesn't appear to be any pattern. It may be related to the institution they are kept in. Otherwise, the format is MM/DD/YYYY.

The "weight" column is often rounded to the nearest 5 lbs.

Statistics for hair:
  43305 Black
  17371 Brown
   2887 Blonde or Strawberry
   2539 Gray or Partially Gray
    740 Red or Auburn
    624 Bald
    396 Not Available
    209 Salt and Pepper
     70 White
      7 Sandy
      1 Unknown

Statistics for sex:
  63409 Male
   4740 Female

Statistics for race:
  37991 Black
  20992 White
   8637 Hispanic
    235 Asian
    104 Amer Indian
     94 Unknown
     92 Bi-Racial
      4 

Statistics for eyes:
  51714 Brown
   7808 Blue
   4259 Hazel
   2469 Green
   1382 Black
    420 Not Available
     87 Gray
      9 Maroon
      1 Unknown



