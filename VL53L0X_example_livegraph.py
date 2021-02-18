import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import VL53L0X

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

xarr = []
yarr = []
count = 0


def animate(i):
    global count
    distance = tof.get_distance()
    count = count + 1
    xarr.append(count)
    yarr.append(distance)
    time.sleep(timing/1000000.00)
    ax1.clear()
    ax1.plot(xarr, yarr)


# Create a VL53L0X object
tof = VL53L0X.VL53L0X()
tof.open()
# Start ranging
tof.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BETTER)

timing = tof.get_timing()
if timing < 20000:
    timing = 20000
print("Timing %d ms" % (timing/1000))

print("Press ctrl-c to exit")

ani = animation.FuncAnimation(fig, animate, interval=100)
plt.savefig("record.png")
plt.show()
tof.stop_ranging()
tof.close()
