Проект по семантической сегментацией, с кастомной реализацией архитектур и сравнением лоссов -  

**bce_loss  VS  dice_loss  VS  focal_loss  VS  custom_loss**

--------------------------------------------


* кастомная имплементация SegNet и UNet с нуля. (метрика - IoU)




для различных лоссов проведено сравнение и сделаны выводы о том, что лучше использовать

* два варианта имплементации U-net

    1. Downsample with MaxPool

       Upsample - nn.Upsample

    2. Downsample with no MaxPool, only Conv layers

       Upsample - ConvTranspose2d

* проведено сравнение как лоссов внутри самих моделей, так и между моделями
