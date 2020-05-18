/* Parameters */
let nTrials = null;
let stimPresentationTime = 30;
let catchPresentationTime = 1000;
let catchFreq = 5;
let rootPath = "https://roi-disruption.s3.amazonaws.com/loc_disruption/";
// let rootPath = "";

/* Globals */
var trials = [];
var curTrial = 0;
var curResponse = null;
var nTraining;
var responseOptionValues;
var trialStartTime;
var experimentStartTime;
var training = true;
var canProceed = true;
var stimPresenting = false;

/* Responses */
var responses = [];
var displayedImages = [];
var types = [];
var trueCats = [];
var foilCats = [];
var reactionTimes = [];

function trialDone() {
    if (!training) {
        // Record the response
        responses.push(curResponse);

        // Record what stimuli were displayed
        trial = trials[curTrial];
        displayedImages.push(trial["image"]);
        types.push(trial["type"]);
        trueCats.push(trial["trueCat"]);
        foilCats.push(trial["foilCat"]);

        var trialEndTime = new Date();
        var rt = trialEndTime - trialStartTime;
        reactionTimes.push(rt);
    }

    if (curTrial === nTraining - 1) {
        training = false;
    }

    curTrial++;

    // Finished experiment
    if (curTrial >= trials.length) {
        doneExperiment();
        return;
    }

    curResponse = null;
    trialBegin();
}

function trialBegin() {
    trialStartTime = new Date();

    // Prepare the stimulus data
    loadTrialData();

    // Pick presentation time based on if this is a catch trial or not
    var presentationTime = stimPresentationTime;
    if (trials[curTrial]["type"] == "catch") {
        presentationTime = catchPresentationTime;
    }

    // Present stimulus
    stimPresenting = true;
    $('#fixation').show();                      // Show fixation cross
    setTimeout(function () {            // Wait briefly before stimulus presentation
        $('#fixation').hide();                  // Hide fixation cross
        $('#trialImage').show();                // Show stimulus
        setTimeout(function () {        // Wait briefly before hiding stimulus
            $('#trialImage').hide();            // Hide stimulus
            setTimeout(function () {    // Wait briefly before presenting response options
                $('#trialOptions').show();      // Present response options
                stimPresenting = false;         // Permit responses
            }, 100);
        }, presentationTime);
    }, 1000);
}

function loadTrialData() {
    trial = trials[curTrial];

    // Set the image
    $('#trialImage').attr("src", trial["imageData"].src);

    // Randomly assign the response options to true/foil answers
    let trueOrder = Math.random() < 0.5;
    responseOptionValues = trueOrder ? [1, 0] : [0, 1];
    var trueText = trueOrder ? $('#option1') : $('#option2');
    var foilText = trueOrder ? $('#option2') : $('#option1');
    trueText.html(trial["trueCat"]);
    foilText.html(trial["foilCat"]);
}

function finishedTraining() {
    canProceed = false;
    $('#trainEndWarning').show();
    $('#proceedExperiment').click(function () {
        canProceed = true;
        $('#trainEndWarning').hide();
        $('#nextTrialMessage').show();
    });
}

function doneExperiment() {
    exportData();
    $("#trial").hide();
    $(document).unbind("keydown.responded");
    $(document).unbind("keydown.nextTrial");
    $("#submitButton").show();
}

function startExperiment() {
    experimentStartTime = new Date();
    $('#startExperiment').hide();
    $('#instructionsContainer').hide();
    $('#trial').show();

    // Click events

    // User has selected a response (pressed a key)
    $(document).bind("keydown.responded", function (event) {
        // Check if the key corresponds to a valid response
        if ((event.which != 70 && event.which != 74) || stimPresenting) {
            return;
        }

        // If this is the last training image, give a warning that must be acknowledged before continuing
        if (curTrial === nTraining - 1 && curResponse === null) {
            finishedTraining();
        }

        // Allow user to continue to the next trial
        if (canProceed) {
            $('#nextTrialMessage').show();
        }

        // Register which response was made
        if (event.which == 70) {
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
        // (and that they've acknowledged being done training)
        if (event.which == 32 && curResponse != null && canProceed) {
            $('#nextTrialMessage').hide();
            $('#trialOptions').hide();
            $('#option1box').css("background-color", "white");
            $('#option2box').css("background-color", "white");
            if (curTrial === nTraining - 1) {                   // If training has ended
                $("#sessionMode").html("Experiment segment")
            }
            trialDone();
        }
    });

    trialBegin();
}

function exportData() {
    $('#displayedImages').val(displayedImages.join());
    $('#types').val(types.join());
    $('#trueCats').val(trueCats.join());
    $('#foilCats').val(foilCats.join());
    $('#responses').val(responses.join());
    $('#reactionTimes').val(reactionTimes.join());
    var curTime = new Date();
    var experimentTime = curTime - experimentStartTime;
    $('#experimentTime').val(experimentTime);
}

/* Setup/preloading code */
function getTrials(callback) {
    $.getJSON(rootPath + "assets/stimuli.json", function (data) {
        var catchStimuli = data["catchStimuli"];
        var trainStimuli = data["trainStimuli"];
        var stimuli = shuffle(data["stimuli"]);

        nTraining = trainStimuli.length;
        if (nTrials == null) {
            nTrials = stimuli.length;
        }

        for (var iStim = 0; iStim < catchStimuli.length; iStim++) {
            var stimulus = catchStimuli[iStim];
            stimulus["type"] = "catch";
        }
        shuffle(catchStimuli);

        for (var iStim = 0; iStim < nTraining; iStim++) {
            var stimulus = trainStimuli[iStim];
            stimulus["type"] = "training";
        }
        shuffle(trainStimuli);

        let types = ["disrupted", "random", "original"];
        for (var iStim = 0, iType = 0; iStim < nTrials; iStim++, iType++) {
            var stimulus = stimuli[iStim];
            stimulus["type"] = types[iType % types.length];
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

function sample(o) {
    return o[Math.floor(Math.random() * o.length)];
}
