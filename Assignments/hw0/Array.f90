! PROGRAM main
! PROGRAM main
!   IMPLICIT NONE

!   INTEGER, DIMENSION(:, :), ALLOCATABLE :: array

!   ALLOCATE (array(2, 3))

!   array = transpose(reshape((/ 1, 2, 3, 4, 5, 6 /),                            &
!     (/ size(array, 2), size(array, 1) /)))

!   print *, array

!   DEALLOCATE (array)

! END PROGRAM main


! !real:: x(3) = (/1,2,3/)
!   integer, dimension(3, 3) :: array,array1
!   array = reshape((/ 1, 2, 3, 4, 5, 6, 7, 8, 9 /), shape(array))
!   array1 = reshape((/ 10, 20, 30, 40, 50, 60, 70, 80, 90 /), shape(array))
  
!   !array = reshape((/ 1, 2, 3, 4, 5, 6, 7, 8, 9 /), shape(array))
!   !real:: y(3,3) = reshape((/1,2,3,4,5,6,7,8,9/), (/3,3/))
!   !integer:: i(3,2,2) = reshape((/1,2,3,4,5,6,7,8,9,10,11,12/), (/3,2,2/))
!   print *,array(2,2)*array1(1,1) !,i

    PROGRAM main
    IMPLICIT NONE

    ! REAL, DIMENSION(10) :: A
    ! REAL, DIMENSION(4,4) :: B
    ! INTEGER i, j

    ! DO i = 1, 10
    !   A(i) = 100 + i
    ! END DO

    ! DO i = 1, 4
	   !    DO j = 1, 4
    !         B(i, j) = j + i + 1
	   !    END DO
    ! END DO


    ! print '("Row form of A"/(10F7.2))', (A(i), i = 1, 10)
    ! print *

    ! print '("Table form of A"/(5F8.2) )', (A(i), i = 1, 10)
    ! print *

    ! print '("Matrix B"/(4F8.2) )', ((B(i, j), i = 1, 4), j = 1, 4)
     real :: r(5,5), d(2,2)
     CALL RANDOM_NUMBER(r)

     CALL RANDOM_NUMBER(d)
     print *, r
     print *
     print *,d


	END
