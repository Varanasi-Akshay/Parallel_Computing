program HW0
  use smoothing !,only: initialize,count
  
  implicit none

  real :: a=0.05,b=0.1,c=0.4 ! constants
  real :: t=0.1,time ! threshold  
  integer :: i1,i2,j,istat
  integer :: n= 16386,nb=0 ! number of elements in each direction
  real,allocatable,dimension(:,:) :: x,y !

  call system_clock(i1,j)
  allocate(x(0:n-1,0:n-1), stat=istat)
  call system_clock(i2,j)
  print *, "The time taken to allocate x is" , real(i2-i1)/real(j)


  call system_clock(i1,j)
  allocate(y(0:n-1,0:n-1), stat=istat)
  call system_clock(i2,j)
  print *, "The time taken to allocate y is" , real(i2-i1)/real(j)


  !call system_clock(count=i1,count_rate=j,count_max=time_max)
  call system_clock(i1,j)
  call initialize(x,n)
  call system_clock(i2,j)
  print *, "The time taken to initialize x is" , real(i2-i1)/real(j)

  call system_clock(i1,j)
  call smooth(x,y,n,a,b,c)
  call system_clock(i2,j)
  !time=(real(i1))-real(i2))/real(j)
  print *, "The time taken to initialize y is" , real(i2-i1)/real(j)


  call system_clock(i1,j)
  call count(x,n,t,nb)
  call system_clock(i2,j)
  print *, "The time taken to count x is" , real(i2-i1)/real(j)


  
  call system_clock(i1,j)
  call count(y,n,t,nb)
  call system_clock(i2,j)
  !time=(real(i1))-real(i2))/real(j)
  print *, "The time taken to count y is" , real(i2-i1)/real(j)


  print *,n
  print *,n-2
  print *,n*n
  print *,(n-2)*(n-2)
  ! print *, 
  ! print *,
  ! print *,
  ! print *,

end program HW0



 