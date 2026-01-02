(* AMFinder - ui/uIDrawing.ml
 *
 * MIT License
 * Copyright (c) 2021 Edouard Evangelisti
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *)

type drawing_tools = {
  surface : Cairo.Surface.t;
  cairo : Cairo.context;
}

module type PARAMS = sig
  val packing : GObj.widget -> unit
end

module type S = sig
  val area : GMisc.drawing_area
  val cairo : unit -> Cairo.context
  val width : unit -> int
  val height : unit -> int
  val synchronize : unit -> unit
  val snapshot : unit -> GdkPixbuf.pixbuf
end

module Make (P : PARAMS) : S = struct
  let area = 
    let frame = GBin.frame ~width:600 ~packing:P.packing () in
    GMisc.drawing_area ~packing:frame#add ()

  let dt = ref None

  let get f () = match !dt with None -> assert false | Some dt -> f dt
  let cairo = get (fun dt -> dt.cairo)
  let width = get (fun dt -> Cairo.Image.get_width dt.surface)
  let height = get (fun dt -> Cairo.Image.get_height dt.surface)

  let synchronize () =
    area#misc#queue_draw ()

  let _ =
    (* Repaint masked area upon GtkDrawingArea exposure. *)
    let draw cr =
      match !dt with
      | None -> false
      | Some { surface; _ } ->
        Cairo.set_source_surface cr surface ~x:0.0 ~y:0.0;
        Cairo.paint cr;
        false
    in
    (* GTK3 draw signal provides cairo context via misc#connect *)
    area#misc#connect#draw ~callback:draw |> ignore;
    (* Create image surface and cairo context on size allocate. *)
    let initialize {Gtk.width; height} =
      let surface = Cairo.Image.(create ARGB32 ~w:width ~h:height) in
      let cairo = Cairo.create surface in
      dt := Some { surface; cairo }
    in
    area#misc#connect#size_allocate ~callback:initialize |> ignore

  let snapshot () =
    match !dt with
    | None -> GdkPixbuf.create ~width:0 ~height:0 ()
    | Some { surface; _ } ->
      let tmp = Filename.temp_file "amf-snap" ".png" in
      Cairo.PNG.write surface tmp;
      let pix = GdkPixbuf.from_file tmp in
      (try Sys.remove tmp with _ -> ());
      pix
end
