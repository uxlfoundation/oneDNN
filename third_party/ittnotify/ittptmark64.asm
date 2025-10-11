COMMENT @
  Copyright (C) 2017-2025 Intel Corporation

  SPDX-License-Identifier: GPL-2.0-only OR BSD-3-Clause
@

;;; ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; ;;; Intel Processor Trace Marker Functionality
;;; ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


_TEXT	segment

        public	__itt_pt_mark
        public	__itt_pt_event
        public	__itt_pt_mark_event
        public  __itt_pt_mark_threshold
        public  __itt_pt_write
        public  __itt_pt_byte

        align   10h

;;; void __itt_pt_mark(unsigned char index);

__itt_pt_mark   proc

        and     rcx,0ffh
        lea     r10,offset __itt_pt_mark_call_table
        lea     rcx,[r10 + rcx * 4]
        jmp	rcx

        align   04h

        dd      0, 1, 2, 3      ;;; GUID

        dd      0fadefadeh      ;;; magic marker

__itt_pt_mark_call_table:

        dd      256 dup(0000c2c3h OR (( $ - offset __itt_pt_mark_call_table) SHL 14))
        
__itt_pt_mark   endp

        align   10h

__itt_pt_byte   proc

        and     rcx,0ffh
        lea     r10,offset __itt_pt_byte_call_table
        add     rcx,r10
        jmp     rcx

        align   04h

        dd      0, 1, 2, 3      ;;; GUID

        dd      0fadedeafh      ;;; magic marker

__itt_pt_byte_call_table:

        db      256 dup(0c3h)

__itt_pt_byte   endp

        align   10h

__itt_pt_event  proc

        rdpmc

        mov     cl,al
        call    __itt_pt_byte
        mov     cl,ah
        call    __itt_pt_byte
        shr     eax,16
        mov     cl,al
        call    __itt_pt_byte
        mov     cl,ah
        call    __itt_pt_byte

        mov     cl,dl
        call    __itt_pt_byte
        mov     cl,dh
        call    __itt_pt_byte
        shr     edx,16
        mov     cl,dl
        call    __itt_pt_byte
        mov     cl,dh
        call    __itt_pt_byte

        ret

__itt_pt_event  endp

        align   10h

__itt_pt_mark_event     proc

        test    rcx,1
        jnz     odd
        mov     r8,rcx
        xor     rcx,rcx
        call    __itt_pt_event
        mov     rcx,r8
        jmp     __itt_pt_mark

odd:
        call    __itt_pt_mark
        xor     rcx,rcx
        jmp     __itt_pt_event

__itt_pt_mark_event     endp

        align   10h

__itt_pt_flush  proc

        lea     rax,offset __itt_pt_mark_flush_1
        jmp     rax

        align   10h
        nop
__itt_pt_mark_flush_1:
        lea     rax,offset __itt_pt_mark_flush_2
        jmp     rax

        align   10h
        nop
        nop
__itt_pt_mark_flush_2:
        lea     rax,offset __itt_pt_mark_flush_3
        jmp     rax

        align   10h
        nop
        nop
        nop
__itt_pt_mark_flush_3:
        ret

__itt_pt_flush  endp

        align   10h

;;; int __itt_pt_mark_threshold(unsigned char index, unsigned long long* tmp, int threshold);

__itt_pt_mark_threshold proc
        mov     r9,rcx  ;;; index
        mov     r10,rdx ;;; tmp
        xor     rdx,rdx
        xor     rax,rax
        test    rcx,1
        jnz     mark_end
mark_begin:
        mov     rcx,(1 SHL 30) + 1
        rdpmc
        shl     rdx,32
        or      rdx,rax
        mov     [r10],rdx
        mov     rcx,r9
        jmp     __itt_pt_mark
mark_end:
        mov     rcx,(1 SHL 30) + 1
        rdpmc
        shl     rdx,32
        or      rdx,rax
        sub     rdx,[r10]
        cmp     rdx,r8  ;;; threshold
        mov     rcx,r9
        jnc     found
        jmp     __itt_pt_mark
found:
        call    __itt_pt_mark
        jmp     __itt_pt_flush

__itt_pt_mark_threshold endp

;;; PTWRITE

        align   10h

;;; void __itt_pt_write(unsigned long long value);

        dd      0, 1, 2, 3      ;;; GUID

__itt_pt_write  proc

;;;        ptwrite rcx
        db      0F3h, 48h, 0Fh, 0AEh, 0E1h
        ret

__itt_pt_write  endp

;;;

_TEXT	ends
        end
