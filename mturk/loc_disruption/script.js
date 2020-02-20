/* Parameters */
let nTrials = null;
let stimPresentationTime = 30;
let catchPresentationTime = 1000;
let catchFreq = 5;
// let rootPath = "https://roi-disruption.s3.amazonaws.com/loc_disruption/";
let rootPath = "";

/* Globals */
var trials = [];
var curTrial = 0;
var nTraining;
var responseOptionValues;
var trialStartTime;
var training = true;
var canProceed = true;

/* Responses */
var trialResponses = [];
var displayedImages = [];
var types = [];
var trueCats = [];
var foilCats = [];
var reactionTimes = [];

function trialDone() {
  var trialEndTime = new Date();
  var rt = trialEndTime - trialStartTime;
  if (!training) {
    reactionTimes.push(rt);
  }

  curTrial++;

  // Finished experiment
  if (curTrial >= nTrials) {
    exportData();
    $('#trial').hide();
    $('#submitButton').show();
    return;
  }

  // Check if training is done
  if (curTrial == nTraining) {
    finishedTraining();
  }

  // Wait for the signal to continue to the next trial
  $('#nextTrialMessage').show();
  $(document).bind("keydown.nextTrial", function(event) {
    if (event.which == 32 && canProceed) {
      $(document).unbind("keydown.nextTrial");
      $('#trialOptions').hide();
      $('#nextTrialMessage').hide();
      $('#option1box').css("background-color", "white");
      $('#option2box').css("background-color", "white");
      if (curTrial == nTraining) {
        $('#sessionMode').html("Experiment segment")
      }
      trialBegin(curTrial);
      trialStartTime = new Date();
    }
  });
}

function trialBegin(trialNum) {
  // Prepare the stimulus data
  loadTrialData(trialNum);

  // Pick presentation time based on if this is a catch trial or not
  var presentationTime = stimPresentationTime;
  if (trials[trialNum]["type"] == "catch") {
    presentationTime = catchPresentationTime;
  }

  // Present stimulus
  $('#fixation').show();                      // Show fixation cross
  setTimeout(function() {             // Wait briefly before stimulus presentation
    $('#fixation').hide();
    $('#trialImage').show();                  // Show stimulus
    setTimeout(function() {           // Wait briefly before hiding stimulus
      $('#trialImage').hide();                // Hide stimulus
      setTimeout(function() {         // Wait briefly before presenting response options
        $('#trialOptions').show();            // Present response options
      }, 100);
    }, presentationTime);

    // Wait for response
    $(document).bind("keydown.response", function(event) {
      if (event.which == 68) {
        $(document).unbind("keydown.response");
        $('#option1box').css("background-color", "lightgrey");
        if (!training) {
          trialResponses.push(responseOptionValues[0]);
        }
        trialDone();
      }
      if (event.which == 75) {
        $(document).unbind("keydown.response");
        $('#option2box').css("background-color", "lightgrey");
        if (!training) {
          trialResponses.push(responseOptionValues[1]);
        }
        trialDone();
      }
    });
  }, 1000);
}

function loadTrialData(trialNum) {
  trial = trials[trialNum];

  // Set the image
  $('#trialImage').attr("src", trial["imageData"].src);

  // Randomly assign the response options to true/foil answers
  let trueOrder = Math.random() < 0.5;
  responseOptionValues = trueOrder ? [1, 0] : [0, 1];
  var trueText = trueOrder ? $('#option1') : $('#option2');
  var foilText = trueOrder ? $('#option2') : $('#option1');
  trueText.html(trial["trueCat"]);
  foilText.html(trial["foilCat"]);

  // Set the results that need to be saved
  if (!training) {
    displayedImages.push(trial["image"]);
    types.push(trial["type"]);
    trueCats.push(trial["trueCat"]);
    foilCats.push(trial["foilCat"]);
  }
}

function finishedTraining() {
  canProceed = false;
  $('#trainEndWarning').show();
  training = false;
  $('#proceedExperiment').click(function () {
    canProceed = true;
    $('#trainEndWarning').hide();
  });
}

function startExperiment() {
  $('#startExperiment').hide();
  $('#instructionsContainer').hide();
  $('#trial').show();
  trialBegin(0);
  trialStartTime = new Date();
}

function exportData() {
  $('#displayedImages').val(displayedImages.join());
  $('#types').val(types.join());
  $('#trueCats').val(trueCats.join());
  $('#foilCats').val(foilCats.join());
  $('#trialResponses').val(trialResponses.join());
  $('#reactionTimes').val(reactionTimes.join());
}

/* Setup/preloading code */
function getTrials(callback) {
  $.getJSON(rootPath + "assets/stimuli.json", function(data) {
    var catchStimuli = data["catchStimuli"];
    var trainStimuli = data["trainStimuli"];
    var stimuli = shuffle(data["stimuli"]);
    var foilCats = data["foilCategories"];

    nTraining = trainStimuli.length;
    if (nTrials == null) {
      nTrials = stimuli.length;
    }

    for (var iStim = 0; iStim < catchStimuli.length; iStim++) {
      var stimulus = catchStimuli[iStim];
      stimulus["type"] = "catch";
      stimulus["foilCat"] = sample(foilCats);
    }
    shuffle(catchStimuli);

    for (var iStim = 0; iStim < nTraining; iStim++) {
      var stimulus = trainStimuli[iStim];
      stimulus["type"] = "training";
      stimulus["foilCat"] = sample(foilCats);
    }
    shuffle(trainStimuli);

    let types = ["disrupted", "random", "original"];
    for (var iStim = 0, iType = 0; iStim < nTrials; iStim++, iType++) {
      var stimulus = stimuli[iStim];
      stimulus["type"] = types[iType % types.length];
      stimulus["foilCat"] = sample(foilCats);
    }
    stimuli = shuffle(stimuli);

    var trialsWithCatches = [];
    for (var iStim = 0, iCatch = 0; iStim < stimuli.length; iStim++) {
      trialsWithCatches.push(stimuli[iStim]);
      if (iStim % catchFreq == catchFreq - 1) {
        trialsWithCatches.push(catchStimuli[iCatch]);
        iCatch++;
      }
    }

    trials = trainStimuli.concat(trialsWithCatches);
    nTrials = trials.length;

    callback();
  });
}

var imgCounter = 0;

function preloadStimuli(callback) {
  for (var i = 0; i < trials.length; i++) {
    preloadImg(trials[i])
  }
  waitForStimuliToPreload(callback);
  console.log('Image preloading complete.');
}

function preloadImg(trial) {
  let imagePath = rootPath + "images/" + trial["type"] + "/" + trial["image"];
  loadImage(imagePath).then((img) => {
      console.log("Preloading:", img);
      trial['imageData'] = img;
      imgCounter++;
      console.log('Image preloading progress: ' + Math.round(100 * (imgCounter / trials.length)) + '%');
  });
}

function loadImage(src) {
    return new Promise((resolve, reject)=> {
        var img = new Image();
        img.onload = ()=> resolve(img);
        img.src = src;
    });
}

function waitForStimuliToPreload(callback) {
  if (imgCounter < trials.length) {
      setTimeout(function() {waitForStimuliToPreload(callback)}, 24);
  } else {
      // load trial
      callback();
  }
}

$(document).ready(function() {
  $('#submitButton').hide();
  getTrials(function() {
    preloadStimuli(function(){
      $('#consent').click(function(){
          $('#startExperiment').click(function(){
              startExperiment();
          });
      });
  });
  });
});

/* Utility functions */
function shuffle(o){
  for(var j, x, i = o.length; i; j = Math.floor(Math.random() * i), x = o[--i], o[i] = o[j], o[j] = x);
  return o;
}

function sample(o) {
  return o[Math.floor(Math.random() * o.length)];
}