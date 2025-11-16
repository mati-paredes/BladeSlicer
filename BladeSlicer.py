import numpy as np
import trimesh
import matplotlib.pyplot as plt
import os
import csv
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import shutil

# ==============================
# Función para procesar STL
# ==============================
def procesar_perfiles():
    global perfiles_originales, checks_vars, carpeta_salida, carpeta_rotados, resultados

    input_stl = filedialog.askopenfilename(
        title="Selecciona archivo STL",
        filetypes=[("Archivos STL", "*.stl"), ("Todos los archivos", "*.*")]
    )
    if not input_stl:
        messagebox.showwarning("Atención", "No se seleccionó ningún archivo STL.")
        return

    stl_name = os.path.splitext(os.path.basename(input_stl))[0]
    base_folder = filedialog.askdirectory(title="Selecciona carpeta de salida")
    if not base_folder:
        messagebox.showwarning("Atención", "No se seleccionó carpeta de salida.")
        return

    stl_folder = os.path.join(base_folder, stl_name)
    os.makedirs(stl_folder, exist_ok=True)

    # -------------------
    # Calcular plane_positions según modo
    # -------------------
    if modo_var.get() == "rango":
        try:
            inicio = float(entry_inicio.get())
            fin = float(entry_fin.get())
            paso = float(entry_paso.get())
            if paso <= 0:
                raise ValueError
            # eje negativo → multiplicar por -1
            plane_positions = np.arange(-inicio, -fin - 1e-6, -paso)
        except ValueError:
            messagebox.showerror("Error", "Valores de inicio, fin o paso inválidos.")
            return
    elif modo_var.get() == "manual":
        try:
            valores = [float(x.strip()) for x in entry_manual.get().split(",") if x.strip() != ""]
            if not valores:
                raise ValueError
            # eje negativo → multiplicar por -1
            plane_positions = np.array([-v for v in valores], dtype=int)
        except ValueError:
            messagebox.showerror("Error", "Formato incorrecto en las distancias manuales.")
            return
    else:
        messagebox.showerror("Error", "Selecciona un modo de entrada válido.")
        return

    # Normalizar: orden descendente (de -inicio hacia -fin), quitar duplicados y asegurar np.array
    plane_positions = np.array(sorted(set(plane_positions), reverse=True), dtype=float)

    carpeta_salida = os.path.join(stl_folder, "perfiles")
    carpeta_rotados = os.path.join(stl_folder, "perfiles_rotados")
    os.makedirs(carpeta_salida, exist_ok=True)
    os.makedirs(carpeta_rotados, exist_ok=True)

    mesh = trimesh.load_mesh(input_stl)
    perfiles_originales = []

    for x_pos in plane_positions:
        slice = mesh.section(plane_origin=[x_pos, 0, 0], plane_normal=[-1, 0, 0])
        if slice is None:
            print(f"No hay intersección en x={x_pos}")
            continue
        slice_2D, transform = slice.to_planar()
        max_path = max(slice_2D.entities, key=lambda p: len(p.points))
        coords = slice_2D.vertices[max_path.points]
        coords[:, 0] = -coords[:, 0]
        coords = coords[:, [1, 0]]

        # Formateo del nombre: sin “p”, sin ceros innecesarios
        fname_pos = str(abs(x_pos)).rstrip("0").rstrip(".")
        filename = os.path.join(carpeta_salida, f"perfil_{fname_pos}.dat")
        with open(filename, "w") as f:
            f.write(f"# Perfil a x={abs(x_pos)} mm\n")
            np.savetxt(f, coords, fmt="%.6f")

        perfiles_originales.append((coords[:, 0], coords[:, 1], -int(x_pos)))


    # ordenar archivos por la posición numérica (revirtiendo el 'p' si corresponde)
    def parse_nombre(n):
        base = os.path.basename(n).replace("perfil_", "").replace(".dat", "")
        return float(base)


    archivos = sorted(
        glob.glob(os.path.join(carpeta_salida, "*.dat")),
        key=parse_nombre
    )

    resultados = []
    for archivo in archivos:
        r_mm = parse_nombre(archivo)
        data = np.loadtxt(archivo, comments="#")
        y = data[:, 0]
        z = data[:, 1]
        puntos = np.column_stack((y, z))
        dist_max = 0
        p1, p2 = None, None
        for i in range(len(puntos)):
            for j in range(i + 1, len(puntos)):
                d = np.linalg.norm(puntos[i] - puntos[j])
                if d > dist_max:
                    dist_max = d
                    p1, p2 = puntos[i], puntos[j]
        if p1 is None or p2 is None:
            continue
        dy = abs(p2[0] - p1[0])
        dz = abs(p2[1] - p1[1])
        theta_rad = np.arctan2(dz, dy) if dy != 0 else np.pi/2
        theta_deg = np.degrees(theta_rad)
        resultados.append([r_mm, round(dist_max, 6), round(theta_rad, 5), round(theta_deg, 2),
                           round(p1[0], 6), round(p1[1], 6), round(p2[0], 6), round(p2[1], 6)])

    archivo_csv = os.path.join(stl_folder, "CHORD_THETA.csv")
    with open(archivo_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["r_mm", "cuerda_mm", "theta_rad", "theta_deg",
                         "p1_x", "p1_y", "p2_x", "p2_y"])
        writer.writerows(resultados)

    chord_theta = {r: (theta, cuerda) for r, cuerda, theta, _, _, _, _, _ in resultados}

    for nombre in os.listdir(carpeta_salida):
        if not nombre.endswith(".dat"):
            continue
        r_mm = float(nombre.replace("perfil_", "").replace(".dat", "").replace('p', '.'))
        if r_mm not in chord_theta:
            continue
        theta_rad, cuerda_real = chord_theta[r_mm]
        data = np.loadtxt(os.path.join(carpeta_salida, nombre), comments="#")
        R = np.array([[np.cos(theta_rad), np.sin(theta_rad)],
                      [-np.sin(theta_rad), np.cos(theta_rad)]])
        puntos_rot = data @ R.T
        p1 = puntos_rot[np.argmin(puntos_rot[:, 0])]
        p2 = puntos_rot[np.argmax(puntos_rot[:, 0])]
        puntos_rot -= p1
        puntos_norm = puntos_rot / np.linalg.norm(p2 - p1)
        idx_te = np.argmin(np.abs(puntos_norm[:, 0] - 1.0))
        puntos_ajustados = np.roll(puntos_norm, -idx_te, axis=0)
        salida = os.path.join(carpeta_rotados, nombre)
        with open(salida, "w") as f:
            f.write(f"# Perfil {r_mm}\n")
            np.savetxt(f, puntos_ajustados, fmt="%.6f")

    for widget in frame_lista_interior.winfo_children():
        widget.destroy()
    checks_vars.clear()
    for _, _, pid in perfiles_originales:
        var = tk.BooleanVar(value=False)
        chk = ttk.Checkbutton(frame_lista_interior, text=f"Perfil {abs(pid)}", variable=var)
        chk.pack(fill="x", padx=5, pady=2)
        checks_vars.append(var)

    boton_graficar_todos.state(["!disabled"])
    boton_graficar_sel.state(["!disabled"])
    boton_graficar_normalizados.state(["!disabled"])
    boton_marcar_todos.state(["!disabled"])
    messagebox.showinfo("Proceso completado",
                        f"Perfiles: {len(perfiles_originales)}\nCSV: {archivo_csv}\nRotados en: {carpeta_rotados}")


# ==============================
# Función de graficado
# ==============================
def graficar(indices=None, normalizados=False):
    if not perfiles_originales:
        messagebox.showwarning("Sin datos", "No hay perfiles cargados")
        return
    if indices is None:
        indices = list(range(len(perfiles_originales)))
    elif not indices:
        messagebox.showwarning("Selección vacía", "No se seleccionó ningún perfil")
        return

    plt.figure(figsize=(8, 5))
    for i in indices:
        pid = int(perfiles_originales[i][2])
        if normalizados:
            archivo = os.path.join(carpeta_rotados, f"perfil_{pid}.dat")
            if not os.path.exists(archivo):
                continue
            data = np.loadtxt(archivo, comments="#")
            plt.plot(data[:, 0], data[:, 1], label=f"Perfil {pid}")
        else:
            x, y, _ = perfiles_originales[i]
            plt.plot(x, y, label=f"Perfil {pid}")
            if mostrar_puntos.get():
                csv_path = os.path.join(os.path.dirname(carpeta_salida), "CHORD_THETA.csv")
                if os.path.exists(csv_path):
                    with open(csv_path, newline="") as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            try:
                                if abs(float(row["r_mm"]) - pid) < 1e-6:
                                    p1x, p1y = float(row["p1_x"]), float(row["p1_y"])
                                    p2x, p2y = float(row["p2_x"]), float(row["p2_y"])
                                    plt.scatter([p1x, p2x], [p1y, p2y], color='red', marker='o', zorder=5)
                                    plt.plot([p1x, p2x], [p1y, p2y], color='red', linestyle='--', linewidth=1)
                                    break
                            except Exception:
                                continue

    plt.title("Perfiles Normalizados" if normalizados else "Perfiles Originales")
    plt.xlabel("X [cuerda normalizada]" if normalizados else "Y [mm]")
    plt.ylabel("Y [cuerda normalizada]" if normalizados else "Z [mm]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

def graficar_seleccionados(normalizados=False):
    seleccion = [i for i, var in enumerate(checks_vars) if var.get()]
    graficar(seleccion, normalizados)

def graficar_todos(normalizados=False):
    graficar(normalizados=normalizados)

def comparar_perfiles_externos():
    if not perfiles_originales:
        messagebox.showwarning("Sin datos", "Primero procesa un STL para obtener perfiles.")
        return

    seleccion = [i for i, var in enumerate(checks_vars) if var.get()]
    if not seleccion:
        messagebox.showwarning("Selección vacía", "Selecciona al menos un perfil para comparar.")
        return

    archivos = filedialog.askopenfilenames(
        title="Selecciona perfiles normalizados externos (.dat)",
        filetypes=[("Archivos DAT", "*.dat"), ("Todos los archivos", "*.*")]
    )
    if not archivos:
        return

    plt.figure(figsize=(8, 5))
    for i in seleccion:
        _, _, pid = perfiles_originales[i]
        archivo_local = os.path.join(carpeta_rotados, f"perfil_{pid}.dat")
        if os.path.exists(archivo_local):
            data_local = np.loadtxt(archivo_local, comments="#")
            plt.plot(data_local[:, 0], data_local[:, 1], label=f"STL perfil {pid}", linewidth=1.8)

    for archivo in archivos:
        try:
            data_ext = np.loadtxt(archivo, comments="#")
            nombre = os.path.basename(archivo)
            plt.plot(data_ext[:, 0], data_ext[:, 1], '--', label=f"Externo: {nombre}", linewidth=1.2)
        except Exception as e:
            print(f"Error al leer {archivo}: {e}")
            continue

    plt.title("Comparación: Perfiles seleccionados vs Externos")
    plt.xlabel("X [cuerda normalizada]")
    plt.ylabel("Y [cuerda normalizada]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

def marcar_todos():
    for var in checks_vars:
        var.set(True)

# ==============================
# Interfaz Tkinter (MODERNA)
# ==============================
root = tk.Tk()
root.title("Procesador de perfiles STL")
root.geometry("640x700")
root.configure(bg="#f3f6fa")

style = ttk.Style()
style.configure("TButton", font=("Segoe UI", 10), padding=6)
style.configure("TLabel", font=("Segoe UI", 10), background="#f3f6fa")
style.configure("TCheckbutton", background="#f3f6fa", font=("Segoe UI", 10))

ttk.Label(root, text="Procesador de Perfiles STL", font=("Segoe UI", 16, "bold")).pack(pady=15)

frame_rango = ttk.Frame(root)
frame_rango.pack(pady=10)
for i, text in enumerate(["Inicio (mm):", "Fin (mm):", "Paso (mm):"]):
    ttk.Label(frame_rango, text=text).grid(row=0, column=2*i, padx=5)
entry_inicio = ttk.Entry(frame_rango, width=10); entry_inicio.grid(row=0, column=1)
entry_fin = ttk.Entry(frame_rango, width=10); entry_fin.grid(row=0, column=3)
entry_paso = ttk.Entry(frame_rango, width=10); entry_paso.grid(row=0, column=5)


# ==============================
# Selección de modo de entrada
# ==============================
modo_var = tk.StringVar(value="rango")

frame_modo = ttk.Frame(root)
frame_modo.pack(pady=5)

ttk.Label(frame_modo, text="Modo de selección de planos:").grid(row=0, column=0, columnspan=2, sticky="w")

ttk.Radiobutton(frame_modo, text="Usar rango automático", variable=modo_var, value="rango").grid(row=1, column=0, sticky="w", padx=10)
ttk.Radiobutton(frame_modo, text="Usar distancias manuales", variable=modo_var, value="manual").grid(row=1, column=1, sticky="w", padx=10)

frame_manual = ttk.Frame(root)
frame_manual.pack(pady=5)
ttk.Label(frame_manual, text="Distancias manuales (mm, separadas por comas):").pack(anchor="w", padx=5)
entry_manual = ttk.Entry(frame_manual, width=60)
entry_manual.pack(padx=5, pady=3)


ttk.Button(root, text="Seleccionar STL y procesar", command=procesar_perfiles).pack(pady=10)

frame_lista = ttk.Frame(root)
frame_lista.pack(pady=10, fill="both", expand=True)
ttk.Label(frame_lista, text="Perfiles detectados:", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=5)

canvas = tk.Canvas(frame_lista, bg="#ffffff", highlightthickness=1, highlightbackground="#d0d0d0")
scrollbar = ttk.Scrollbar(frame_lista, orient="vertical", command=canvas.yview)
frame_lista_interior = ttk.Frame(canvas)
frame_lista_interior.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=frame_lista_interior, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

frame_buttons = ttk.Frame(root)
frame_buttons.pack(pady=15)

boton_graficar_sel = ttk.Button(frame_buttons, text="Graficar seleccionados",
                                command=lambda: graficar_seleccionados(normalizados=False))
boton_graficar_sel.grid(row=0, column=0, padx=5)

boton_graficar_todos = ttk.Button(frame_buttons, text="Graficar todos",
                                  command=lambda: graficar_todos(normalizados=False))
boton_graficar_todos.grid(row=0, column=1, padx=5)

boton_graficar_normalizados = ttk.Button(frame_buttons, text="Graficar normalizados",
                                         command=lambda: graficar_seleccionados(normalizados=True))
boton_graficar_normalizados.grid(row=0, column=2, padx=5)

boton_marcar_todos = ttk.Button(frame_buttons, text="Marcar todos", command=marcar_todos)
boton_marcar_todos.grid(row=0, column=3, padx=5)

boton_comparar_externos = ttk.Button(frame_buttons, text="Comparar con externos",
                                     command=comparar_perfiles_externos)
boton_comparar_externos.grid(row=0, column=4, padx=5)

mostrar_puntos = tk.BooleanVar(value=False)
ttk.Checkbutton(root, text="Mostrar puntos y cuerda", variable=mostrar_puntos).pack(pady=5)

ttk.Button(root, text="Salir", command=root.destroy).pack(pady=10)

perfiles_originales = []
checks_vars = []
carpeta_salida = ""
carpeta_rotados = ""

root.mainloop()
