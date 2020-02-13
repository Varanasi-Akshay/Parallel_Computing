module smoothing
implicit none
contains
    subroutine initialize(array,num)
    !implicit none
    integer :: num
    real,dimension(num,num) :: array
    call random_number(array)
    return
    end subroutine initialize
 
!Smoothing function
!contains
    subroutine smooth(x,y,num,a,b,c)
    !implicit none
    integer :: i,j,num
    real :: a,b,c
    real,dimension(num,num) :: x,y ! 
    do i=1,num
      do j=1,num
         y(i,j)= 0 !a*(x(i-1,j-1)+x(i-1,j+1)+x(i+1,j-1)+x(i+1,j+1))+ b*(x(i-1,j+0)+x(i+1,j+0)+x(i+0,j-1)+x(i+0,j+1))+c*(x(i+0,j+0))
      enddo
    enddo

   ! do i=2,num-1
   !   do j=2,num-1
   !     y(i,j)=a*(x(i-1,j-1)+x(i-1,j+1)+x(i+1,j-1)+x(i+1,j+1))+ b*(x(i-1,j+0)+x(i+1,j+0)+x(i+0,j-1)+x(i+0,j+1))+c*(x(i+0,j+0))
   !   enddo
   ! enddo

    return

    end subroutine smooth

! Counting the 
    subroutine count(array,num,threshold,num_below) ! takes array, number of elements in one direction, threshold and number of elements below threshold
    implicit none
    integer :: i,j,num
    integer,intent(out)::num_below
    real :: array(num,num)
    real :: threshold
    do i=1,num
   	  do j=1,num

         if(array(i,j)< threshold) then
          num_below=num_below+1 
         endif 
      enddo  
    enddo
    return
    end subroutine count
end module smoothing
