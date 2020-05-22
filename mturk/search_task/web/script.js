let debug = true;
let hideInstructions = false;

/* Parameters */
let rootPath = "https://roi-manipulation.s3.amazonaws.com/searchtask/";
if (debug) {
    rootPath = "";
}
let targetTimeTrain = 2000;
let targetTimeExp = 500;
let fixationTime = 500;
let feedbackTimeTrain = 2000;
let feedbackTimeExp = 1000;
let numHitSets = 3;

/* Globals */
var nTrials = null;
var trials = [];
var imageLocations;
var curTrial = 0;
var response;
var reactionTime;
var nTraining;
var trialStartTime;
var experimentStartTime;
var training = true;

/* Responses */
var trialImages = [];
var rois = [];
var reactionTimes = [];
var responses = [];
var corrects = [];

function trialDone() {
    if (!training) {
        reactionTimes.push(reactionTime);
        responses.push(response);
        corrects.push(response === trials[curTrial]["answer"] ? 1 : 0)
        trialImages.push(trials[curTrial]["image"]);
        rois.push(trials[curTrial]["roi"]);
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
        $("#trainEndWarning").show();
        $(document).bind("keydown.proceed", function (event) {
            if (event.which === 32) {
                $(document).unbind("keydown.proceed");
                $("#trainEndWarning").hide();
                trialBegin();
            }
        });
    } else {
        trialBegin();
    }
}

function trialBegin() {
    $('#targetImage').attr("src", trials[curTrial]["targetImageData"].src);
    $('#trialImage').attr("src", trials[curTrial]["trialImageData"].src);

    var targetTime = targetTimeExp;
    if (training) {
        targetTime = targetTimeTrain;
    }

    $("#targetImage").show();
    setTimeout(function () {
        $("#targetImage").hide();
        $("#fixation").show();
        setTimeout(function() {
            $("#fixation").hide();
            $("#trialImage").show();
            showTrial();
        }, fixationTime);
    }, targetTime);
}

function showTrial() {
    // Wait for user to find the target image
    var displayedTime = new Date();
    $(document).bind("keydown.found", function (event) {
        if (event.which === 32) {
            $(document).unbind("keydown.found");
            reactionTime = new Date() - displayedTime;
            $("#trialImage").hide();
            $("#selection").show();
        }
    });
}

function responseCallback(responseVal) {
    return function () {
        $("#selection").hide();
        response = responseVal;
        showFeedback(trialDone);
    };
}

function showFeedback(callback) {
    var answer = trials[curTrial]["answer"];
    if (answer === response) {
        $("#incorrectHighlight").hide();
        $("#feedbackText").html("Correct!");
        $("#feedbackText").css("color", "green");
    } else {
        $("#incorrectHighlight").show();
        $("#feedbackText").html("Incorrect.");
        $("#feedbackText").css("color", "red");
    }

    placeHighlight = function (highlight, loc) {
        var l = loc[0];
        var t = loc[1];
        var w = loc[2] - loc[0];
        var h = loc[3] - loc[1];
        var borderWidth = parseInt(highlight.css("border-left-width"));
        highlight.css({left: l - borderWidth, top: t - borderWidth, width: w, height: h});
    };
    placeHighlight($("#correctHighlight"), imageLocations[answer]);
    placeHighlight($("#incorrectHighlight"), imageLocations[response]);

    var feedbackTime = feedbackTimeExp;
    if (training) {
        feedbackTime = feedbackTimeTrain;
    }

    $("#trialImage").show();
    $("#feedbackHighlights").show();
    $("#feedbackText").show()
    setTimeout(function() {
        $("#trialImage").hide();
        $("#feedbackHighlights").hide();
        $("#feedbackText").hide()
        callback();
    }, feedbackTime);
}

function startExperiment() {
    experimentStartTime = new Date();
    $('#startExperiment').hide();
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
    $('#trialImages').val(trialImages.join());
    $('#rois').val(rois.join());
    $('#reactionTimes').val(reactionTimes.join());
    $('#responses').val(responses.join());
    $('#corrects').val(corrects.join());
    var curTime = new Date();
    var experimentTime = curTime - experimentStartTime;
    $('#experimentTime').val(experimentTime);
    $("#hitSetId").val(hitSetId);
}

/* Setup/preloading code */
function getTrials(callback) {
    hitSetId = getRandomInt(0, numHitSets - 1);
    $.getJSON(rootPath + "assets/trialData.json", function (data) {
        var trainTrials = shuffle(data["trainTrials"][hitSetId]);
        var expTrials = shuffle(data["experimentTrials"][hitSetId]);
        nTraining = trainTrials.length;
        trials = trainTrials.concat(expTrials);
        nTrials = trials.length;
        callback();
    });
}

function makeSelectionButtons(callback) {
    $.getJSON(rootPath + "assets/locations.json", function (data) {
        // Get locations of images within a trial and rescale them based on displayed size
        var locations = data["locations"];
        var sizeRatio = $("#trialImage").width() / trials[0]["trialImageData"].width;
        for (i = 0; i < locations.length; i++) {
            var loc = locations[i];
            var l = loc[0] * sizeRatio;
            var t = loc[1] * sizeRatio;
            var r = l + (loc[2] - loc[0]) * sizeRatio;
            var b = t + (loc[3] - loc[1]) * sizeRatio;
            locations[i] = [l, t, r, b];
        }
        imageLocations = locations;

        // Create selection buttons at each location that set the value of the current response
        for (i = 0; i < locations.length; i++) {
            var loc = locations[i];
            var l = loc[0];
            var t = loc[1];
            var w = loc[2] - loc[0];
            var h = loc[3] - loc[1];

            var button = $("<div>", {"id": "selectionButton" + i, "class": "selectionButton"});
            button.css({left: l, top: t, width: w, height: h});
            button.click(responseCallback(i));
            $("#selection").append(button);
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
    let trialImagePath = rootPath + "assets/images/" + trial["image"] + ".jpg";
    loadImage(trialImagePath).then((img) => {
        console.log("Preloading:", img);
        trial['trialImageData'] = img;
        imgCounter++;
        console.log('Image preloading progress: ' + Math.round(100 * (imgCounter / (2 * trials.length))) + '%');
    });
    let targetImagePath = rootPath + "assets/images/" + trial["image"] + "_target.jpg";
    loadImage(targetImagePath).then((img) => {
        console.log("Preloading:", img);
        trial['targetImageData'] = img;
        imgCounter++;
        console.log('Image preloading progress: ' + Math.round(100 * (imgCounter / (2 * trials.length))) + '%');
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
    $("#trial").hide();
    getTrials(function () {
        preloadStimuli(function () {
            makeSelectionButtons(function () {
                if (!hideInstructions) {
                    $("#startExperiment").click(function () {
                        if ($("#consent").prop("checked") === false) {
                            return;
                        }
                        startExperiment();
                    });
                } else{
                    startExperiment();
                }
            });
        });
    });
});

/* Utility functions */
function shuffle(o) {
    for (var j, x, i = o.length; i; j = Math.floor(Math.random() * i), x = o[--i], o[i] = o[j], o[j] = x) ;
    return o;
}

function getRandomInt(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min + 1)) + min;
}
