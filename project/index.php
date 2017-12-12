<!doctype html>
<html lang="en">
  <head>
    <title>Deep Learning Project by Nitin Tokas</title>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta.2/css/bootstrap.min.css" integrity="sha384-PsH8R72JQ3SOdhVi3uxftmaW6Vc51MKb0q5P2rRUpPvrszuE4W1povHYgTpBfshb" crossorigin="anonymous">
  </head>
  <body>
    <nav class="navbar navbar-expand-lg navbar-light bg-light">
      <a class="navbar-brand" href="#">Deep Learning Project</a>
      <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarText" aria-controls="navbarText" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
      </button>
      <div class="collapse navbar-collapse" id="navbarText">
        <ul class="navbar-nav mr-auto">
          <li class="nav-item">
            <a class="nav-link" style="display:none;" href="#">Home <span class="sr-only"></span></a>
          </li>
          <li class="nav-item">
            <a class="nav-link" style="display:none;" href="#">Details</a>
          </li>
          
        </ul>
        <span class="navbar-text">
          Nitin Tokas #U01372189
        </span>
      </div>
    </nav>

    <div class="container" style="margin-top:5%;margin-bottom:5%;">

      <h1>Hello, User! Welcome to Image recognizition using Keras.</h1>
      
      <hr>
      <h2>CAMERA BLOCK</h2>
      <hr>
    <div class="row" style="min-height:350px;">

      <div class="col-md-5">
        
        <video id="video" style="margin-top:-15%;" width="400px" height="400px" autoplay></video>
      </div>
      <div class="col-md-2">

        <button class="btn snaps btn-outline-success" id="snap1">Snap Group 1 Photo</button>
        <br><br>
        <button class="btn snaps btn-outline-success" id="snap2">Snap Group 2 Photo</button>
        <br><br>
        <button class="btn snaps btn-outline-success" id="snap3">Snap Group 3 Photo</button>
        <br><br>
        <button class="btn btn-outline-primary" id="sim_model" style="display:none;">Simulate model</button>
        <button class="btn btn-outline-primary" id="result" style="display:none;">Find Result</button>
        <br>
        <div id="message"></div>
      </div>
      <div class="col-md-5">
        <p>Welcome to my project page of Deep Learning. Here, we are going to do a catergorial learning using keras and tensorflow. First click some snaps as different groups. Then these pictures will be used by our model to learn these different sets of images to differentiate between them. After completion of this step we will then click one more picture and predict what does this picture refer to.</p>
      </div>

    </div>
    <div class="row">

      <div class="col-md-12">
        <div id="current_result" style="display:none;"></div>
      </div>

    </div>
    <div class="row">

      <div class="col-md-4" id="group1" style="display:none;">
        <h2>Group 1 Photo</h2>
        <canvas id="canvas1" width="350" height="350"></canvas>
      </div>
      <div class="col-md-4" id="group2" style="display:none;">
        <h2>Group 2 Photo</h2>
        <canvas id="canvas2" width="350" height="350"></canvas>
      </div>
      <div class="col-md-4" id="group3" style="display:none;">
        <h2>Group 3 Photo</h2>
        <canvas id="canvas3" width="350" height="350"></canvas>
      </div>

    </div>
    <div id="log_data"></div>
    
    


    </div>
    <!-- Optional JavaScript -->
    <!-- jQuery first, then Popper.js, then Bootstrap JS -->
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.3/umd/popper.min.js" integrity="sha384-vFJXuSJphROIrBnz7yo7oB41mKfc8JzQZiCq4NCceLEaO4IHwicKwpJf9c9IpFgh" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta.2/js/bootstrap.min.js" integrity="sha384-alpBpkh1PFOepccYVYDB4do5UnbKysX5WZXm3XxPqe5iKTfUKjNkCk9SaVuEZflJ" crossorigin="anonymous"></script>

     <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>

    <script type="text/javascript">
    var video = document.getElementById('video');

    // Get access to the camera!
    if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        // Not adding `{ audio: true }` since we only want video now
        navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
            video.src = window.URL.createObjectURL(stream);
            video.play();
        });
    }
    var canvas1 = document.getElementById('canvas1');
    var canvas2 = document.getElementById('canvas2');
    var canvas3 = document.getElementById('canvas3');
    var context1 = canvas1.getContext('2d');
    var context2 = canvas2.getContext('2d');
    var context3 = canvas3.getContext('2d');
    var x,x1,x2,x3=0;
    // Trigger photo take
    document.getElementById("snap1").addEventListener("click", function() {
      $("#group1").show();
      context1.drawImage(video, 0, 0, 350, 350);
      $(".snaps").hide();

      $("#message").html("Generating Group 1 images....");
      $.get("get_screenshots.php?x=1", function(data, status){
        $(".snaps").show();
        $("#message").html("");
        x1=1;
        x=x1+x2+x3;
        model(x);
      });
    });
    document.getElementById("snap2").addEventListener("click", function() {
      $("#group2").show();
      context2.drawImage(video, 0, 0, 350, 350);
      $(".snaps").hide();
      $("#message").html("Generating Group 2 images....");
      $.get("get_screenshots.php?x=2", function(data, status){
        $(".snaps").show();
        $("#message").html("");
        x2=1;
        x=x1+x2+x3;
        model(x);
      });
    });
    document.getElementById("snap3").addEventListener("click", function() {
      $("#group3").show();
      context3.drawImage(video, 0, 0, 350, 350);
      $(".snaps").hide();
      $("#message").html("Generating Group 3 images....");
      $.get("get_screenshots.php?x=3", function(data, status){
        $(".snaps").show();
        $("#message").html("");
        x3=1;
        x=x1+x2+x3;
        model(x);
      });
    });
    function model(x){

      if(x>2){
        $("#sim_model").show();
      }
    }
    document.getElementById("sim_model").addEventListener("click", function() {
      $(".snaps").hide();
      $("#sim_model").hide();
      $("#message").html("Generating Model....");
      $.get("get_model.php?x=1", function(data, status){
        //$(".snaps").show();
        $("#message").html("");
        //$("#sim_model").show();

        $("#log_data").html("<pre>"+data+"</pre>");

        $("#result").show();
      });
    });

    document.getElementById("result").addEventListener("click", function() {
      result();
    });

    function result() {
      $("#message").html("Finding Result....");
      $("#result").hide();
      $.get("get_model.php?x=2", function(data, status){
        $("#result").show();
        $("#message").html("");
        $("#current_result").html("<p>"+data+"</p>");
        $("#current_result").show();
        //result();
      });
    }
    </script>

  </body>
</html>