# Intersection over Union for object detection

The inputs are two sets of boxes, each containing _n_ and _m_ sets of corner coordinates, respectively. The output is an _n_ x _m_ matrix containing IoU values for each pairing of boxes from the two sets.

## Description

The inspiration for this mini-project was roughly drawn from an assignment specifying some interesting requirements. For instance, use numpy and no explicit control flow or loop constructs (`for`, `while`, `if`...). It seemed fun enough. Full assignment description is available here: https://github.com/photomath/ml-assignments/blob/main/assignment-B.pdf.

## Usage

You can import or copy/paste functions from [iou.py](https://github.com/MarkoDuksi/Intersection-over-union/blob/main/iou.py) and use them in your project or tweak the code and run it to do some benchmarks. Here is just an example:

```
>>>  python iou.py
sqrt(num_boxes);time/s
200;0.0034
400;0.0103
600;0.0222
800;0.0410
1000;0.0642
1200;0.0959
1400;0.1291
1600;0.1694
1800;0.2179
2000;0.2588
2200;0.3099
2400;0.3678
2600;0.4307
2800;0.5025
3000;0.6260
3200;0.6928
3400;0.7942
3600;0.8891
3800;0.9938
4000;1.0734
4200;1.2104
4400;1.3213
4600;1.4648
4800;1.5533
5000;1.7194
```

### Comments on the results

 A nested loops algorithm was implemented as a baseline for comparison to two vectorized solutions. Initial benchmarks (like the one in the above example) were done on a single core of Intel® Core™ i7-3770K CPU @ 3.50GHz with 16 GB RAM. Operating system was Debian 11 and no particular system optimizations were made.

 All algorithms exhibited _O_(_n_ x _m_) time complexity as expected. Nested loops algorithm needs the least amount of memory but is extremely slow. Vectorized algorithms need more memory but are more than 200 times faster. Both vectorized solutions exhibit an upward bend in the slope at a point where a constant overhead is introduced as the system starts running low on memory. On this particular system the runtimes of vectorized solutions scale linearly up to about _n_ x _m_ = 1.5e8 before the overhead somewhat steepens the slope. See the charts to better understand this observation.


![IoU until no more RAM available](https://github.com/MarkoDuksi/Intersection-over-union/blob/main/images/Chart_1.png)
**Chart 1.** IoU until no more RAM available

![IoU before the system overhead](https://github.com/MarkoDuksi/Intersection-over-union/blob/main/images/Chart_2.png)
**Chart 2.** IoU before the system overhead

 Interestingly, both vectorized solutions were equally agnostic to output sparsity. The final benchmark was done using Google Colab and _n_ x _m_ = 2000 x 2000 (for both dense and sparse case) to check if the solutions meet the sub-1-second Google Colab runtime goal. Runtimes below 0.5 seconds are demonstrated in the [notebook](https://github.com/MarkoDuksi/Intersection-over-union/blob/main/notebooks/IoU.ipynb) along with some additional exploration.

## Improvements proposal

Embarrassingly parallel solution is obvious but not pursued.

## Authors

Marko Dukši
[@LinkedIn](https://www.linkedin.com/in/mduksi/)

## Version History

- 0.1
    * Initial Release

## License

This project is licensed under the MIT License.
