# engr421-homework-5-multivariate-parametric-classification-solved
**TO GET THIS SOLUTION VISIT:** [ENGR421 Homework 5-Multivariate parametric classification Solved](https://www.ankitcodinghub.com/product/engr-421-dasc-521-introduction-to-machine-learning-solved-5/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;113746&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;ENGR421 Homework 5-Multivariate parametric classification Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
In this homework, you will implement a multivariate parametric classification algorithm using Python. Here are the steps you need to follow:

1. Read Chapter 5 from the textbook.

2. Generate random data points from four bivariate Gaussian densities with the following parameters:

𝝁! = #++04..05) , 𝚺! = #++30..20 +0.0) ,

+1.2 𝑁! = 105

𝝁” = #−−41..50) , 𝚺” = #++10..28 +0.8) ,

+1.2 𝑁” = 145

𝝁# = #+−41..50) , 𝚺# = #+−10..28 −0.8) ,

+1.2 𝑁# = 135

𝝁$ = #+−04..00) , 𝚺$ = #++10..20 +0.0) ,

+3.2 𝑁$ = 115

Your data points should be like the following figure. (10 points)

3. Estimate the parameters 𝝁2!, 𝝁2″, 𝝁2#, 𝝁2$, 𝚺3!, 𝚺3″, 𝚺3#, 𝚺3$, 𝑃5(𝑦 = 1), 𝑃5(𝑦 = 2), 𝑃5(𝑦 = 3), and 𝑃5(𝑦 = 4) using the data points you generated in the previous step. Your parameter estimations should be like the following figures. (30 points)

print(sample_means)

[[-2.43085714e-04 4.41475305e+00]

[-4.40159367e+00 -1.00817799e+00]

[ 4.53185568e+00 -9.79534452e-01]

[-3.20267739e-02 -3.79497784e+00]]

print(sample_covariances) [[[ 3.46382957 0.26022464]

[ 0.26022464 1.19547019]]

[[ 1.34545849 0.78772458]

[ 0.78772458 1.11187005]]

[[ 1.27229804 -0.66903494]

[-0.66903494 0.96283015]]

[[ 1.44282286 -0.20544896]

[-0.20544896 3.2734625 ]]]

print(class_priors) [0.21 0.29 0.27 0.23]

4. Calculate the confusion matrix for the data points in your training set using the parametric classification rule you will develop using the estimated parameters from the previous step. Your confusion matrix should be like the following matrix. (30 points)

print(confusion_matrix) y_truth 1 2 3 4 y_pred 1 104 1 1 0

2 1 144 0 0

3 0 0 133 0

4 0 0 1 115

5. Draw your decision boundaries that you will calculate using the parametric classification rule from the previous step together with data points and clearly mark misclassified data points. Your figure should be like the following figure. (30 points)

What to submit: You need to submit your source code in a single file (.py file) named as STUDENTID.py, where STUDENTID should be replaced with your 7-digit student number.

How to submit: Submit the file you created to Blackboard. Please follow the exact style mentioned and do not send a file named as STUDENTID.py. Submissions that do not follow these guidelines will not be graded.

Cheating policy: Very similar submissions will not be graded.
