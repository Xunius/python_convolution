module CONV1D
implicit none

private
integer, parameter :: ikind = selected_real_kind(p=15,r=15)

public CONVOLVE1D, RUNMEAN1D

contains

    !--------Do 1d convolution on 2d slab with given kernel--------
    subroutine CONVOLVE1D(slab,slabmask,kernel,kernelflag,max_missing,resslab,resmask,hs,ws)
        ! 1D convolution on 2d slab with given kernel

        ! <slab>: real, 2d input array. Convolution will be done on its 1st dimension.
        ! <slabmask>: int, 2d mask for <slab>, 0 means valid, 1 means missing/invalid/nan.
        ! <kernel>: real, 1d input array with smaller size than <slab>, kernel to convolve with.
        ! <kernelflag>: int, 1d flag for <kernel>, 0 means empty, 1 means something. This is 
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

        integer, parameter :: ikind = selected_real_kind(p=15,r=15)

        integer :: hs, ws
        real(kind=ikind), dimension(hs,ws), intent(in) :: slab
        real(kind=ikind), dimension(hs,ws), intent(out) :: resslab
        integer, dimension(hs,ws), intent(in) :: slabmask
        integer, dimension(hs,ws), intent(out) :: resmask
        real(kind=ikind), dimension(:), intent(inout) :: kernel
        integer, dimension(:), intent(inout) :: kernelflag
        real, intent(in) :: max_missing

        integer :: lk, ii, jj, i, nvalid, nbox
        integer :: hh_u, hh_d, y1, y2
        real(kind=ikind) :: tmp

        !--------Default values--------
        resslab=0.
        resmask=1

        !-------------------Check shape-------------------
        lk=size(kernel)   ! length of kernel

        if (lk /= size(kernelflag)) then
            write(*,*) 'Length do not match between <kernel> and <kernelflag>.'
            return
        end if

        if (lk >= hs) then
            write(*,*) 'kernel size too large.'
            return
        end if

        if (max_missing <=0 .OR. max_missing >1) then
            write(*,*) '<max_missing> needs to be in range (0,1].'
            return
        end if

        !-------------------Flip kernel-------------------
        kernel=kernel(ubound(kernel,1):lbound(kernel,1):-1)
        kernelflag=kernelflag(ubound(kernelflag,1):lbound(kernelflag,1):-1)

        !---------------Half kernel lengths---------------
        if (mod(lk,2) == 1) then
            hh_u=(lk-1)/2
            hh_d=(lk-1)/2
        else
            hh_u=lk/2-1
            hh_d=lk/2
        end if

        !-------------------Loop through columns-------------------
        do jj = 1,ws
            !-----------------convolve column-----------------
            do ii = 1,hs

                y1=ii-hh_u
                y2=ii+hh_d

                tmp=0.
                nvalid=0
                nbox=0

                do i = y1,y2
                    ! skip out-of-bound points
                    if (i<1 .or. i>hs) then
                        cycle
                    end if
                    nvalid=nvalid+(1-slabmask(i,jj))*kernelflag(i-y1+1)
                    tmp=tmp+(1-slabmask(i,jj))*slab(i,jj)*kernel(i-y1+1)
                    nbox=nbox+kernelflag(i-y1+1)
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

    end subroutine CONVOLVE1D


    !------------Compute 1D moving average on 2d data------------
    subroutine RUNMEAN1D(slab,slabmask,kernel,kernelflag,max_missing,resslab,resmask,hs,ws)
        ! Compute 1D moving average on given axis of 2d data

        ! <slab>: real, 2d input array. Running mean will be done on its 1st dimension.
        ! <slabmask>: int, 2d mask for <slab>, 0 means valid, 1 means missing/invalid/nan.
        ! <kernel>: real, 1d input array with smaller size than <slab>, kernel to convolve with.
        ! <kernelflag>: int, 1d flag for <kernel>, 0 means empty, 1 means something. This is 
        !               to fercilitate counting valid points within element.
        ! <max_missing>: real, max percentage of missing within any convolution element tolerable.
        !                E.g. if <max_missing> is 0.5, if over 50% of values within a given element
        !                are missing, the center will be set as missing (<res>=0, <resmask>=1). If
        !                only 40% is missing, center value will be computed using the remaining 60%
        !                data in the element.
        ! <hs>, <ws>: int, height and width of <slab>. Optional.
        ! 
        ! Return <resslab>: real, 2d array, result of running mean.
        !        <resmask>: int, 2d array, mask of the result. 0 means valid, 1 means missing (no
        !                   enough data within element).

        implicit none

        integer, parameter :: ikind = selected_real_kind(p=15,r=15)

        integer :: hs, ws
        real(kind=ikind), dimension(hs,ws), intent(in) :: slab
        real(kind=ikind), dimension(hs,ws), intent(out) :: resslab
        integer, dimension(hs,ws), intent(in) :: slabmask
        integer, dimension(hs,ws), intent(out) :: resmask
        real(kind=ikind), dimension(:), intent(inout) :: kernel
        integer, dimension(:), intent(inout) :: kernelflag
        real, intent(in) :: max_missing

        real(kind=ikind), dimension(hs,ws) :: num
        real(kind=ikind), dimension(hs,ws) :: den

        !-------------------Flip kernel-------------------
        kernel=kernel(ubound(kernel,1):lbound(kernel,1):-1)
        kernelflag=kernelflag(ubound(kernelflag,1):lbound(kernelflag,1):-1)

        call CONVOLVE1D(slab,slabmask,kernel,kernelflag,max_missing,num,resmask,hs,ws)
        call CONVOLVE1D(1.-real(slabmask,ikind),slabmask,kernel,kernelflag,max_missing,den,resmask,hs,ws)

        resslab=num/den

    end subroutine RUNMEAN1D
end module CONV1D
