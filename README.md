# Character Level LanguageModel
A 3 layer LSTM based character level language model, trained on Charles Dickens books.

Detailed report on results and experiments conducted, visit <a href="https://github.com/sidd-pandey/NUS-MtechKE-ProjectReports/tree/master/Deep%20Learnining"> this link </a>.

<br>
A simple deployment is done using Heroku. Visit the deployed model by following this link: <a href="https://charlesdickens.herokuapp.com/">https://charlesdickens.herokuapp.com/</a><br>
Flask is used to build the API and render the template. Gunicorn is used as web server. Had to decrease the number of workers to 1, to be able to deploy under 512M. <br><br>

<img src = "https://github.com/sidd-pandey/Char-LanguageModel/blob/master/imgs/img-1.PNG" width="600"/>