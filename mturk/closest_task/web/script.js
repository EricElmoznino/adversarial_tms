let debug = false;

/* Parameters */
let rootPath = "https://roi-manipulation.s3.amazonaws.com/closesttask/";
if (debug) {
    rootPath = "";
}

/* Globals */
var nTrials = null;
var trials = [];
var curTrial = 0;
var loadedImages = {};
var nTraining;
var trialStartTime;
var experimentStartTime;
var training = true;

/* Responses */
var trialImages = [];
var responses = [];
var responseButtonIds = [];
var conditions = [];
var reactionTimes = [];

function trialDone(selection) {
    if (!training) {
        trialImages.push(trials[curTrial]["orig"]);
        responses.push(trials[curTrial]["roiOrder"][selection]);
        responseButtonIds.push(selection);
        reactionTimes.push((new Date()) - trialStartTime);
        conditions.push(trials[curTrial]["condition"]);
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

    // Proceed to next trial, pausing for confirmation if we are at the end of training
    if (curTrial === nTraining) {
        $("#trialDisplay").hide();
        $("#trainEndWarning").show();
        $("#endTraining").click(function () {
            $("#trialDisplay").show();
            $("#trainEndWarning").hide();
            trialBegin();
        });
    } else {
        trialBegin();
    }
}

function trialBegin() {
    $('#origImage').attr("src", loadedImages[trials[curTrial]["orig"]].src);
    $('#option1Image').attr("src", loadedImages[trials[curTrial]["gen"][0]].src);
    $('#option2Image').attr("src", loadedImages[trials[curTrial]["gen"][1]].src);
    $('#option3Image').attr("src", loadedImages[trials[curTrial]["gen"][2]].src);
    $('#option4Image').attr("src", loadedImages[trials[curTrial]["gen"][3]].src);
    trialStartTime = new Date();
}

function startExperiment() {
    experimentStartTime = new Date();
    $('#instructionsContainer').hide();
    $("#trial").show();
    trialBegin();
}

function doneExperiment() {
    $("#trial").hide();
    exportData();
    $("#submitButton").show();
}

function exportData() {
    $('#trialImage').val(trialImages.join());
    $('#response').val(responses.join());
    $('#responseButtonId').val(responseButtonIds.join());
    $('#reactionTime').val(reactionTimes.join());
    $('#condition').val(conditions.join());
    var curTime = new Date();
    var experimentTime = curTime - experimentStartTime;
    $('#experimentTime').val(experimentTime);
}

/* Setup/preloading code */
function getTrials(callback) {
    $.getJSON(rootPath + "assets/trialData.json", function (data) {
        var trainTrials = shuffle(data["trainTrials"]);
        var expTrials = shuffle(data["experimentTrials"]);
        nTraining = trainTrials.length;
        trials = trainTrials.concat(expTrials);
        nTrials = trials.length;
        preloadStimuli(callback, data["images"]);
    });
}

var numImages;
var imgCounter = 0;

function preloadStimuli(callback, images) {
    numImages = images.length;
    for (var i = 0; i < numImages; i++) {
        preloadImg(images[i])
    }
    waitForStimuliToPreload(callback);
    console.log('Image preloading complete.');
}

function preloadImg(image) {
    let imagePath = rootPath + "assets/images/" + image;
    loadImage(imagePath).then((img) => {
        console.log("Preloading:", img);
        loadedImages[image] = img;
        imgCounter++;
        console.log('Image preloading progress: ' + Math.round(100 * (imgCounter / numImages)) + '%');
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
    if (imgCounter < numImages) {
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
    $("#trial").hide();
    getTrials(function () {
        $("#startExperiment").click(function () {
            if ($("#consent").prop("checked") === false) {
                return;
            }
            startExperiment();
        });
    });
});

/* Utility functions */
function shuffle(o) {
    for (var j, x, i = o.length; i; j = Math.floor(Math.random() * i), x = o[--i], o[i] = o[j], o[j] = x) ;
    return o;
}
