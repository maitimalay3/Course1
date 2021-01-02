My Targets are -

Your new target is:
99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
Less than or equal to 15 Epochs
Less than 10000 Parameters (additional points for doing this in less than 8000 pts)
Do this in exactly 4 steps
Each File must have "target, result, analysis" TEXT block (either at the start or the end)
You must convince why have you decided that your target should be what you have decided it to be, and your analysis MUST be correct. 
Evaluation is highly subjective, and if you target anything out of the order, marks will be deducted

Step 1:S5_Experimentbatchsize32LRNOSTEP
Target: Prepare the basic structure. Reduce the parameter size from ~20,000 to less than 10,000 and achive the accuracy >99%. Try to keep the parameters less than 8000
Approach: 
	Introduce conv 1*1 to reduce the number channels, before each transition block.
        Use data augmentation (Rotate)
Result:
Used parameters - 7,966
Maximum Train Accuracy Achieved -98.94
Maximum Test Accuracy Achieved - 99.23
Maximum - Epoch -15

Analysis:
There is scope to improve the training accuracy either by changing  model hyperparameters (LR, Batch size etc.) or by by adding additional model parameters (add conv layers)

Step 2:S5_ExperimentLRSTEPBatch32
Target:Increase the test accuracy >99.40
Approach: Change the Learning rate parameter from 05 to 01 and also introduce stepLR

Result:
Used parameters - 7,966
Maximum Train Accuracy Achieved -98.86
Maximum Test Accuracy Achieved - 99.30
Maximum - Epoch -15

Analysis: Training accuracy is still <99.00 and test accuracy at 99.30. Need to introduce additional conv layer to better fit the training data and improve the model accuracy.

        
Step 3: S5_ExperimentLR_STEPBatch16
Target:Increase the test accuracy >99.40. Reduce the gap between training and test accuracy  
Approach: Experimented different batch sizes (64, 32, 16). 
Result:
Used parameters - 7,966
Maximum Train Accuracy Achieved -98.84
Maximum Test Accuracy Achieved - 99.38
Maximum - Epoch -15

Analysis: Accruacy decreases if the batch size is greater than 32. On an average, batch size of 16 and 32 provides similar traininag and test accuracy


Step 4:
Target:Increase the training accuracy above 99%> and the test accuracy >99.40     
Approach: Introduced additional convolution layer, which has increased the model parmeters from 7, 966 to 9,692 

Result:
Used parameters - 9,692
Maximum Train Accuracy Achieved -99.11
Maximum Test Accuracy Achieved - 99.41
Maximum - Epoch -15

Analysis: By introducing additional 2000 parameters, both training and test accuracy has accuracy has marginally increased by 0.1 (test accuracy increased from 99.3 to 99.4)
Training and test accuracies are reasonably close, no sign of over fitting (training accuracy is significantly higher than the test accuracy) so didn't explore the regularization techniques like frop out. etc.

