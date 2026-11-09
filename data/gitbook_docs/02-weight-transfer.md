# Weight Transfer

Understanding the concept of "weight transfer" is the most important element of how car setups and driving technique affect the handling balance of your car.

Perhaps you already understand this, but as you increase the vertical load on a tire that increases the grip that tire will produce. This is why downforce cars have more grip for example. But we can use this concept to affect the balance of the car both in the way the car is setup and also your driving technique.

#### <mark style="color:purple;">**Key Takeaway**</mark>

<mark style="color:purple;">As you increase the vertical load on a tire that increases the grip that tire will produce.</mark>

So first let's understand the concepts of "Lateral" (between left and right side of the car) and "Longitudinal" (between front and back) weight transfer and then explore how we can control those two different weight transfers using setup and driving techniques.

***

### Longitudinal Weight Transfer

Longitudinal weight transfer is more straight forward than Lateral weight transfer but more on that later. Longitudinal weight transfer is about transferring weight between the front and rear tires of the car.

For example, since the more vertical load a tire has the more grip it has, then the more of the car's existing weight can be transferred to the front of the car during corner entry the more steering the car will have on corner entry.

Likewise, if you were to transfer less weight to the front tires on corner entry then you will have less steering (i.e more understeer). Finding the right amount of weight transfer to get the car balance you want is part setup and part driving technique. More on that later.

This is not to say that if you add more physical weight to the front of the car it will steer more, as that would cause the car to steer less due to the increase in centrifugal force by adding more weight to the front. This is why for example the Porsche 911 with the gas tank up front will have more steering as the fuel burns off the car because you are reducing the physical weight at the front of the car.

#### How to affect longitudinal weight transfer with setups

The greater the rake in the car (see ride height section for more info) the more weight is placed on the front tires even at a static state. But keep in mind that it is not a lot more. If you go into iRacing and put the front and rear ride heights to be the same and then look at the corner weights iRacing calculates, and then compare those corner weights after you jack up the rear ride height a lot, you will not see much difference in the weight over the front tires compared to rear tires between These two settings. So, it is a small difference, but it is still there.

Similar results if you soften the front springs and stiffen the rear springs (see springs section for more info). You will see only a very small difference in the weight being added to the front tires and removed from the rear tires in this static state.

Also, with either of these rake or spring changes, when you brake you will be transferring slightly more weight forward than before these changes.

#### <mark style="color:purple;">**Key Takeaway**</mark>

<mark style="color:purple;">Longitudinal weight transfer has a massive impact on car balance, and you can control it via driving technique and setup.</mark>

***

### Lateral Weight Transfer

Lateral weight transfer is the transfer of your car's existing weight between the left side tires and the right side tires. As you turn a corner the inside tires will have less vertical load reducing their grip and the outside tires will have more vertical load increasing their grip. i.e the weight is transferred from the inside to the outside tires. But there is a very important detail about this that when you understand it will help you with fine tuning the balance of your car. So, let's get into it.

#### But how does lateral weight transfer affect the balance of your car?

The important thing to remember about lateral weight transfer is, the greater the difference in load between the inside tire and the outside tire (i.e. more lateral weight transfer), the less overall lateral grip the car will have.

The reason for this is that as we increase the load on the tire by "X" amount, we don't get the same "X" amount increase in lateral grip. The curved line on the "Tire Performance Curve" shows that as you increase the vertical load on a tire by 100 units, the tire does not produce the same incremental 100 units of lateral grip.

So, this means you can affect the balance of the car by having different amounts of weight transfer across the front tires than the weight transfer across the rear tires. Specifically end of the car that has more weight transfer from the inside to the outside tire will have less grip than the other end of the car.

#### <mark style="color:purple;">**Key Takeaway**</mark>

<mark style="color:purple;">As a car transfers weight from the inside tire to the outside tire during cornering the overall lateral grip decreases because the inside tire loose more grip than the outside tire gains in grip. The greater the lateral weight transfer the less lateral grip is generated</mark>

<mark style="color:purple;">This means that by having more lateral weight transfer at one end of the car, that end of the car will have less grip, thus affecting the balance of the car</mark>

#### **Example of equal lateral weight transfer both front and rear**

To further help you understand this, I used the numbers in the tire performance curve shown above and did the math as shown in the table below. These calculations show that if we assume we are turning left, the car has 50/50 front to rear weight distribution, weighs 2000 units and assumes the front and rear tires transfer the same amount of weight from the inside tire to the outside tire.

In this example I assume the lateral weight transfer for both front and rear tires transfers 500 units of vertical load from the inside tire to the outside tire. <mark style="color:purple;">The overall lateral grip is reduced from 1042 units to 925 units (300 for inside tire and 625 for the outside tire), for both the front and the rear. This shows how lateral weight transfer will reduce lateral grip, and this can be leveraged to affect the balance of the car as explained below.</mark>

|                               | Static | While Cornering                        |
| ----------------------------- | ------ | -------------------------------------- |
| Car Weight                    | 2000   | 2000                                   |
| Front weight distribution     | 50%    | 50%                                    |
| Rear weight distribution      | 50%    | 50%                                    |
| Front lateral weight transfer |        | 500                                    |
| Rear lateral weight transfer  |        | 500                                    |
| **FRONT**                     |        |                                        |
| FL vertical load              | 500    | 250                                    |
| FL laterial grip              | 521    | 300                                    |
| FR vertical load              | 500    | 750                                    |
| FR lateral grip               | 521    | 625                                    |
| Total front lateral grip      | 1042   | <mark style="color:purple;">925</mark> |
| Front tire grip efficiency    | 1.04   | 0.925                                  |
| **REAR**                      |        |                                        |
| RL vertical load              | 500    | 250                                    |
| RL lateral grip               | 521    | 300                                    |
| RR vertical load              | 500    | 750                                    |
| RR lateral grip               | 521    | 625                                    |
| Total rear grip               | 1042   | <mark style="color:purple;">925</mark> |
| Rear tire grip efficiency     | 1.04   | 0.925                                  |

#### **Example of more oversteer through less front lateral weight transfer**

By controlling how the amount lateral weight transfer at the front of the car versus the rear of the car or at least how fast this weight transfer happens then we can affect the balance of the car. For example, if you are able to get the car to transfer less lateral load at the front tires (200 units in this example) than the rear tires (500 units in this example) this will give the front tires more lateral grip (1034 units) than the rear tires (925 units), as shown in the calculations below, causing the car to have more steering during this left hand corner.

|                               | Static | While Cornering                          |
| ----------------------------- | ------ | ---------------------------------------- |
| Car Weight                    | 2000   | 2000                                     |
| Front weight distribution     | 50%    | 50%                                      |
| Rear weight distribution      | 50%    | 50%                                      |
| Front lateral weight transfer |        | <mark style="color:purple;">200</mark>   |
| Rear lateral weight transfer  |        | 500                                      |
| **FRONT**                     |        |                                          |
| FL vertical load              | 500    | <mark style="color:purple;">400</mark>   |
| FL laterial grip              | 521    | <mark style="color:purple;">458</mark>   |
| FR vertical load              | 500    | <mark style="color:purple;">600</mark>   |
| FR lateral grip               | 521    | <mark style="color:purple;">576</mark>   |
| Total front lateral grip      | 1042   | <mark style="color:purple;">1034</mark>  |
| Front tire grip efficiency    | 1.04   | <mark style="color:purple;">1.034</mark> |
| **REAR**                      |        |                                          |
| RL vertical load              | 500    | 250                                      |
| RL lateral grip               | 521    | 300                                      |
| RR vertical load              | 500    | 750                                      |
| RR lateral grip               | 521    | 625                                      |
| Total rear grip               | 1042   | 925                                      |
| Rear tire grip efficiency     | 1.04   | 0.925                                    |

### Controlling lateral weight transfer through setups

All of the above information is interesting only if we can affect the amount of lateral weight transfer. So, let's look at how we can do that.

Lateral weight transfer is a function of the weight of the car, the lateral force, the location of the center of gravity and the track width, as shown in the following formula.

*<mark style="color:red;">Lateral Weight Transfer = (CarWeight x Lateral g's x CGHeight)/ Gravity x TrackWidth</mark>*

We can simplify this equation by assuming a lateral g of 1.0 which is equal to gravity, so the formula is now\...

*<mark style="color:red;">LWT = (CarWeight x CGHeight)/ TrackWidth</mark>*

For example:

*<mark style="color:red;">LWT = (1500 Weight Units x 20 Distance Units) / 185 Distance Units = 162 Weight Units</mark>* of total lateral weight transfer assuming the car weights 1500 Weight Units and the CG is at 20 Distance Units above ground.

If we lower the CG 2 Distance Units

*<mark style="color:red;">LWT = (1500 Weight Units x 18 Distance Units) / 185 Distance Units = 145 Weight Units</mark>* of total lateral weight transfer.

#### <mark style="color:purple;">**Key Takeaway**</mark>

<mark style="color:purple;">This means that</mark> <mark style="color:purple;"></mark>*<mark style="color:purple;">raising the CG will provide more lateral weight transfer and thus less lateral grip</mark>*<mark style="color:purple;">, while lowering the CG will provide less lateral weight transfer and thus more lateral grip.</mark>

<mark style="color:purple;">So, we can use the front and rear ride height of the car to raise or lower the CG, which affects lateral weight transfer and thus lateral grip. For example, if we raise the rear ride height this raises the cg of the car at the rear more than the front which increases the lateral weight transfer at the rear, which reduces the grip the tires can generate at the rear.</mark>

<mark style="color:purple;">Also keep in mind that the lower the ride height in general the lower the CG so you will want to run the lowest ride height you can both front and rear while still maintaining the balance you want and not bottoming out the suspension.</mark>

#### Swaybars effect on Lateral Weight Transfer

So, you may have been told that stiffer swaybars at the front of the car will reduce understeer, but you never understood why, and in fact it always felt counter intuitive to you because, well race cars have stiff swaybars compared to road cars so stiff swaybars must produce more grip right? Let me try to explain this to you and a\`lso explain why race cars have stiff swaybars.

As you stiffen swaybars, you might conclude that it reduces the lateral weight transfer due to less body roll. However, because swaybars are designed to push up on the inside tire and down on the outside tire swaybars increase the difference in vertical load between the inside and outside tire, which will decrease the lateral grip.

#### <mark style="color:purple;">**Key Takeaway**</mark>

<mark style="color:purple;">A stiffer swaybar increases lateral weight transfer to that end of the car, which takes away lateral grip from that end of the car.</mark>

So why then do race cars have stiff swaybars. They do this to allow for quick direction changes and to have more consistent handling because it reduces the overall movement of the car.

#### **Sometimes you get opposite results**

Let me use an example of how you can get an opposite result from what I explained above. Let's assume you are at Road America and going through the kink in the Porsche 911 GT3R, and the car is getting too loose for you. Let's assume also that you are running a low wing, so you have a low rear ride height (i.e. less rake) to keep the car stable (see the Ride Height section for more on how lower rear ride height makes the car more stable). So you think hey I need to soften the rear swaybar to reduce the lateral weight transfer in the rear, so the rear has more grip. Go try it,...and you will likely find that if you soften the rear arb too much the rear of the car gets more loose. At this point you have lost all faith in what I am explaining here right ? :) Well with setups you can never just isolate things, as a setup is the sum of its parts. In this scenario by allowing the rear to roll too much you are losing aero grip in the rear because the rear outside gets too low and falls out of the optimal ride height range to maximize aero grip.

#### Roll Center effect on Lateral Weight Transfer

Not many cars in iRacing allow you to control the Roll Center, but the V8 Supercars are one of them that does. Roll Center effect on lateral weight transfer to be honest is a very complicated and somewhat contradictory thing. The reason for this is that both a high RC and a low RC seem to contribute to increasing the lateral weight transfer for different reasons and it is very complex to figure out which approach transfers more lateral weight than the other.

The high RC increases lateral weight transfer to the outside tire because of it's tendency to resist the roll. In contrast the lower RC does not resist the roll as much, so you might think that it transfers less lateral weight transfer, but you need to consider the fact that as the car rolls intuitively this transfers more weight to the outside tires.

In the research I did, the additional weight transfer due to roll was dismissed in the lateral weight transfer calculations as insignificant, because in race cars that don't roll much anyway, it was said that the roll contributes less than 3% to the lateral weight transfer. This suggests that if your car is not rolling much due to other settings like swaybars etc, the lower the roll center the more lateral grip you will have because the lateral weight transfer will be less. However with softer suspensions and swaybars, you may find that a higher roll center will produce more lateral grip.

***

### Driving Technique and how it affects Weight transfer and thus balance

So far, we have talked about how setups can affect weight transfer, so now let's discuss how your driving technique can be used to exploit weight transfer to get the balance you want.

For example, as you are braking you are transferring weight that was on the rear tires to the front tires, giving the front tires more relative grip than the rear tires, assuming you are not locking tires up. If you do let's say lock the front tires up or get into the abs on the front tires (micro locking) then you are not maximizing the grip to the front tires because firstly they are sliding too much which means they are not griping to their full potential, and because of this you can't transfer as much weight to the front tires further reducing the grip the front tires have. This is why if you trail brake too hard into the entry of a corner your car will push, and if you don't brake enough your car will push. There is a sweet spot to be found that optimizes the amount of front grip to achieve the balance in the car you desire all from trail braking.

***

### **So lets summarize**

* Weight transfer is longitudinal and lateral and both affect car balance
* Using lateral weight transfer to loosen the car means you need more lateral weight transfer across the rear tires relative to the front tires. This can be done by stiffening the rear swaybar/spring (or softening the front swaybar/spring) or raising the rear ride height (i.e more rake).
* Using lateral weight transfer to make the car push more, you need more lateral weight transfer across the front tires relative to the rear. This can be done by stiffening the front sway bar (or softening the rear swaybar), or lowering the rear ride height.
* Using Longitudinal weight transfer to loosen the car means transferring more weight to the front tires during trail braking/cornering. This can be done by finding the optimal trail braking so you maximize the weight transfer without locking up the tires or using ABS. You can also increase the rear ride height (i.e. increase the rake) or soften the front springs or stiffen the rear springs to gain slightly more weight transfer forward. The front springs have a bigger impact on trail braking and the rear springs have bigger impact on corner exit acceleration.
* Using Longitudinal weight transfer to make the car push more, means transferring less weight to the front tires during braking/cornering. This can be done by braking too much causing the fronts to lock or get into ABS or by braking less to reduce the weight transfer. You can also decrease the rear ride height (i.e. decrease the rake) or stiffen the front springs or soften the rear springs to gain slightly more weight transfer forward. The front springs have a bigger impact on trail braking and the rear springs have bigger impact on corner exit acceleration.
