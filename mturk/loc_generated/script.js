/* Parameters */
// let rootPath = "https://roi-disruption.s3.amazonaws.com/loc_disruption/";
let rootPath = "";

/* Globals */
var trials = [];
var curTrial = 0;
var curResponse = null;
var responseOptionValues;
var trialStartTime;

/* Responses */
var responses = [];
var trueCats = [];
var foilCats = [];
var displayedImages = [];
var isRandoms = [];
var reactionTimes = [];

function trialDone() {
    // Record the response
    responses.push(curResponse);

    // Record what stimuli that were displayed
    trial = trials[curTrial];
    trueCats.push(trial["trueCat"]);
    foilCats.push(trial["foilCat"]);
    displayedImages.push(trial["image"]);

    // Note whether or not this was a catch trial
    if (trial["type"] === "random") {
        isRandoms.push(1);
    } else {
        isRandoms.push(0);
    }

    // Record the reaction time
    var trialEndTime = new Date();
    var rt = trialEndTime - trialStartTime;
    reactionTimes.push(rt);

    curTrial++;

    // Finished experiment
    if (curTrial >= trials.length) {
        doneExperiment();
        return;
    }

    curResponse = null;
    trialBegin();
}

function trialBegin(trialNum) {
    trialStartTime = new Date();
    $("#trialImage").prop("src", trials[curTrial]["imgData"].src);

    // Randomly assign the response options to true/foil answers
    if (Math.random() < 0.5) {
      responseOptionValues = [trials[curTrial]["trueCat"], trials[curTrial]["foilCat"]];
      $('#option1').html(trials[curTrial]["trueCat"]);
      $('#option2').html(trials[curTrial]["foilCat"]);
    } else {
      responseOptionValues = [trials[curTrial]["foilCat"], trials[curTrial]["trueCat"]];
      $('#option1').html(trials[curTrial]["foilCat"]);
      $('#option2').html(trials[curTrial]["trueCat"]);
    }
}

function doneExperiment() {
    exportData();
    $("#trial").hide();
    $(document).unbind("keydown.responded");
    $(document).unbind("keydown.nextTrial");
    $("#submitButton").show();
}

function startExperiment() {
    $("#startExperiment").hide();
    $("#instructionsContainer").hide();
    $("#trial").show();

    // Click events

    // User has selected a response (pressed a key)
    $(document).bind("keydown.responded", function (event) {
        // Check if the key corresponds to a valid response
        if (event.which != 68 && event.which != 75) {
            return;
        }

        // Allow user to continue to the next trial
        $('#nextTrialMessage').show();

        // Register which response was made
        if (event.which == 68) {
            curResponse = responseOptionValues[0];
            $('#option1box').css("background-color", "lightgrey");
            $('#option2box').css("background-color", "white");
        } else {
            curResponse = responseOptionValues[1];
            $('#option2box').css("background-color", "lightgrey");
            $('#option1box').css("background-color", "white");
        }
    });

    // User wishes to continue to the next trial (pressed the "Space" key)
    $(document).bind("keydown.nextTrial", function (event) {
        // Check if they pressed the space bar and that they've responded
        if (event.which == 32 && curResponse != null) {
            $('#nextTrialMessage').hide();
            $('#option1box').css("background-color", "white");
            $('#option2box').css("background-color", "white");
            trialDone();
        }
    });

    trialBegin();
}

function exportData() {
    $('#response').val(responses.join());
    $('#trueCat').val(trueCats.join());
    $('#foilCat').val(foilCats.join());
    $('#displayedImage').val(displayedImages.join());
    $('#isRandom').val(isRandoms.join());
    $('#reactionTime').val(reactionTimes.join());
}

/* Setup/preloading code */
function getTrials(callback) {
    $.getJSON(rootPath + "assets/stimuli.json", function (data) {
        trials = shuffle(data["stimuli"]);

        for (var i = 0; i < trials.length; i++) {
            if (Math.random() < 0.5) {
              trials[i]["type"] = "targeted";
            } else {
              trials[i]["type"] = "random";
            }
        }

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
        trial['imgData'] = img;
        imgCounter++;
        console.log('Image preloading progress: ' + Math.round(100 * (imgCounter / trials.length)) + '%');
    });
}

function loadImage(src) {
    return new Promise((resolve, reject) => {
        var img = new Image();
        img.onload = () => resolve(img);
        img.src = src;
    });
}

function waitForStimuliToPreload(callback) {
    if (imgCounter < trials.length) {
        setTimeout(function () {
            waitForStimuliToPreload(callback)
        }, 24);
    } else {
        // load trial
        callback();
    }
}

$(document).ready(function () {
    $('#submitButton').hide();
    getTrials(function () {
        preloadStimuli(function () {
            $('#consent').click(function () {
                $('#startExperiment').click(function () {
                    startExperiment();
                });
            });
        });
    });
});

/* Utility functions */
function shuffle(o) {
    for (var j, x, i = o.length; i; j = Math.floor(Math.random() * i), x = o[--i], o[i] = o[j], o[j] = x) ;
    return o;
}
