module CONV2D
implicit none

private
integer, parameter :: ikind = selected_real_kind(p=15,r=10)

public CONVOLVE2D, RUNMEAN2D

contains

    !--------Convolve 2d slab with given kernel--------
    subroutine CONVOLVE2D(slab,slabmask,kernel,kernelflag,max_missing,resslab,resmask,hs,ws)
        ! Convolve 2d slab with given kernel

        ! <slab>: real, 2d input array.
        ! <slabmask>: int, 2d mask for <slab>, 0 means valid, 1 means missing/invalid/nan.
        ! <kernel>: real, 2d input array with smaller size than <slab>, kernel to convolve with.
        ! <kernelflag>: int, 2d flag for <kernel>, 0 means empty, 1 means something. This is 
        !               to fercilitate counting valid points within element.
        ! <max_missing>: real, max percentage of missing within any convolution element tolerable.
        !                E.g. if <max_missing> is 0.5, if over 50% of values within a given element
        !                are missing, the center will be set as missing (<res>=0, <resmask>=1). If
        !                only 40% is missing, center value will be computed using the remaining 60%
        !                data in the element.
        ! <hs>, <ws>: int, height and width of <slab>. Optional.
        ! 
        ! Return <resslab>: real, 2d array, result of convolution.
        !        <resmask>: int, 2d array, mask of the result. 0 means valid, 1 means missing (no
        !                   enough data within element).

        implicit none

        integer, parameter :: ikind = selected_real_kind(p=15,r=10)

        integer :: hs, ws
        real(kind=ikind), dimension(hs,ws), intent(in) :: slab
        real(kind=ikind), dimension(hs,ws), intent(out) :: resslab
        integer, dimension(hs,ws), intent(in) :: slabmask
        integer, dimension(hs,ws), intent(out) :: resmask
        real(kind=ikind), dimension(:,:), intent(inout) :: kernel
        integer, dimension(:,:), intent(inout) :: kernelflag
        real, intent(in) :: max_missing

        integer :: hk, wk, ii, jj, i, j, nvalid, nbox
        integer :: hw_l, hw_r, hh_u, hh_d, x1, x2, y1, y2
        real(kind=ikind) :: tmp

        !--------Default values--------
        resslab=0.
        resmask=1

        !-------------------Check shape-------------------
        hk=size(kernel,1)    ! height of kernel
        wk=size(kernel,2)    ! width of kernel

        if (hk /= size(kernelflag,1) .OR. wk /= size(kernelflag,2)) then
            write(*,*) 'Shape do not match between <kernel> and <kernelflag>.'
            return
        end if

        if ( hk >= hs .OR. wk >= ws) then
            write(*,*) 'kernel too large.'
            return
        end if

        if (max_missing <=0 .OR. max_missing >1) then
            write(*,*) '<max_missing> needs to be in range (0,1].'
            return
        end if

        !-------------------Flip kernel-------------------
        kernel=kernel(ubound(kernel,1):lbound(kernel,1):-1, ubound(kernel,2):lbound(kernel,2):-1)
        kernelflag=kernelflag(ubound(kernelflag,1):lbound(kernelflag,1):-1, &
            & ubound(kernelflag,2):lbound(kernelflag,2):-1)

        !---------------Half kernel lengths---------------
        if (mod(hk,2) == 1) then
            hh_u=(hk-1)/2
            hh_d=(hk-1)/2
        else
            hh_u=hk/2-1
            hh_d=hk/2
        end if

        if (mod(wk,2) == 1) then
            hw_l=(wk-1)/2
            hw_r=(wk-1)/2
        else
            hw_l=wk/2-1
            hw_r=wk/2
        end if

        !-------------------Loop through grids-------------------
        do ii = 1,hs
            do jj = 1,ws

                y1=ii-hh_u
                y2=ii+hh_d
                x1=jj-hw_l
                x2=jj+hw_r

                tmp=0.
                nvalid=0
                nbox=0

                do i = y1,y2
                    do j = x1,x2
                        ! skip out-of-bound points
                        if (i<1 .or. i>hs .or. j<1 .or. j>ws) then
                            cycle
                        end if

                        nvalid=nvalid+(1-slabmask(i,j))*kernelflag(i-y1+1,j-x1+1)
                        tmp=tmp+(1-slabmask(i,j))*slab(i,j)*kernel(i-y1+1,j-x1+1)
                        nbox=nbox+kernelflag(i-y1+1,j-x1+1)
                    end do
                end do

                if (1-float(nvalid)/nbox < max_missing) then
                    resslab(ii,jj)=tmp
                    resmask(ii,jj)=0
                else
                    resslab(ii,jj)=0.
                    resmask(ii,jj)=1
                end if
            end do
        end do

    end subroutine CONVOLVE2D


    !------------Compute 2D moving average------------
    subroutine RUNMEAN2D(slab,slabmask,kernel,kernelflag,max_missing,resslab,resmask,hs,ws)
        ! Compute 2D moving average

        ! <slab>: real, 2d input array.
        ! <slabmask>: int, 2d mask for <slab>, 0 means valid, 1 means missing/invalid/nan.
        ! <kernel>: real, 2d input array with smaller size than <slab>, kernel to convolve with.
        ! <kernelflag>: int, 2d flag for <kernel>, 0 means empty, 1 means something. This is 
        !               to fercilitate counting valid points within element.
        ! <max_missing>: real, max percentage of missing within any convolution element tolerable.
        !                E.g. if <max_missing> is 0.5, if over 50% of values within a given element
        !                are missing, the center will be set as missing (<res>=0, <resmask>=1). If
        !                only 40% is missing, center value will be computed using the remaining 60%
        !                data in the element.
        ! <hs>, <ws>: int, height and width of <slab>. Optional.
        ! 
        ! Return <resslab>: real, 2d array, result of convolution.
        !        <resmask>: int, 2d array, mask of the result. 0 means valid, 1 means missing (no
        !                   enough data within element).

        implicit none

        integer, parameter :: ikind = selected_real_kind(p=15,r=10)

        integer :: hs, ws
        real(kind=ikind), dimension(hs,ws), intent(in) :: slab
        real(kind=ikind), dimension(hs,ws), intent(out) :: resslab
        integer, dimension(hs,ws), intent(in) :: slabmask
        integer, dimension(hs,ws), intent(out) :: resmask
        real(kind=ikind), dimension(:,:), intent(inout) :: kernel
        integer, dimension(:,:), intent(inout) :: kernelflag
        real, intent(in) :: max_missing

        real(kind=ikind), dimension(hs,ws) :: num
        real(kind=ikind), dimension(hs,ws) :: den

        !-------------------Flip kernel-------------------
        kernel=kernel(ubound(kernel,1):lbound(kernel,1):-1, ubound(kernel,2):lbound(kernel,2):-1)
        kernelflag=kernelflag(ubound(kernelflag,1):lbound(kernelflag,1):-1, &
            & ubound(kernelflag,2):lbound(kernelflag,2):-1)

        call CONVOLVE2D(slab,slabmask,kernel,kernelflag,max_missing,num,resmask,hs,ws)
        call CONVOLVE2D(1.-real(slabmask,ikind),slabmask,kernel,kernelflag,max_missing,den,resmask,hs,ws)

        resslab=num/den

    end subroutine RUNMEAN2D
end module CONV2D
