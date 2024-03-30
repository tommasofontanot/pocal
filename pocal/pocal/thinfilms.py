#from pathlib import Path






class pocal:
    
    def __init__(self, prescription,angle_in_deg,min_wave,max_wave,resolution,ref_wave,opt_thick,refinement_type):
        
        
        
        import numpy as np
        import json
        
        self.prescription = np.genfromtxt(prescription,dtype='str')
        self.angle_in_deg = angle_in_deg #user_change, incidence angle
        self.opt_thick = opt_thick #true or false
        self.ref_wave = ref_wave
        self.json_file_path = 'materialLibrary.json'
        self.starting_angle = np.deg2rad(angle_in_deg)
        self.starting_angle_2 = np.deg2rad(angle_in_deg)
        self.admitt = 0.0026544
        self.resolution = resolution
        self.min_wave = min_wave #user puts
        self.max_wave = max_wave + self.resolution #user puts
        self.wave_spacing = np.arange(self.min_wave, self.max_wave, self.resolution)
        self.resol = (max_wave - min_wave)/self.resolution
        self.refinement_type = refinement_type #'transmittance' or 'reflectance' or None
        
        with open(self.json_file_path, 'r') as j:
             self.contents = json.loads(j.read())
        
        self.unique_set = set(self.prescription[:, 0]) #user upload

        self.unique_set_props = []
        #setting up properties of elements in unique_set
        for index,element in enumerate(self.unique_set):
            temp = []
            w, r, c = self.search_from_library(element) #order wavelength, complex, imaginary
            temp.append(str(element))
            temp.append(w)
            temp.append(r)
            temp.append(c)
            self.unique_set_props.append(temp)
            
        self.nr_k_array = []

        for index,element in enumerate(self.unique_set):
            temp = []
            nr,k = self.nr_and_k_generator(element,self.max_wave,self.wave_spacing) #order wavelength, complex, imaginary
            temp.append(str(element))
            temp.append(nr)
            temp.append(k)
            self.nr_k_array.append(temp)
            
            
        self.number_of_layers = np.size(self.prescription[:, 0])

        ########

        self.nr_medium, self.k_medium =   self.search_from_nr_k_generator(self.nr_k_array,self.prescription[0,0]) 
        self.N_medium = self.nr_medium - 1j*self.k_medium
        ########
        if self.opt_thick == True:
            optical_thickness = self.prescription[:, 1]
            layer_thickness = []
            for l in range(self.number_of_layers):
                nr, k =   self.search_from_nr_k_generator(self.nr_k_array,self.prescription[l,0])
                N_ref = (nr)[self.wave_spacing == self.ref_wave][0]
                layer_thickness.append(float(optical_thickness[l])*(self.ref_wave)/N_ref)
            self.prescription[:,1] = np.array(layer_thickness)
        
    ########     
    #this function "search_from_library" takes one input parameter: "material" and returns the wavelength, real and complex parts
    def search_from_library(self,material):
        mat_wav  = []
        mat_real = []
        mat_comp = []
        for j in range(len(self.contents)):
            if self.contents[j]["material"] == material:
                mat_wav = self.contents[j]["wavelength"]
                mat_real = self.contents[j]["real"]
                mat_comp = self.contents[j]["complex"]

        return mat_wav, mat_real, mat_comp

    #helper function
    def search_from_usp(material,arr):
        for element in arr:
            if element[0] == material:
                return element

    #"nr_and_k_generator" function takes 3 params material,max_wave,wave_spacing; returns nr and k arrays; takes care of wavelength issue       
    def nr_and_k_generator(self,material,max_wave,wave_spacing):
        import numpy as np
        from scipy import interpolate
        
        mat_wav, mat_real, mat_comp = self.search_from_library(material)
        material_full = np.array([])
        nr_material_full = np.array([])
        k_material_full = np.array([])

        max_index = np.size(mat_wav)

        if max_wave > max(mat_wav):
            for i in range(int(((max_wave - max(mat_wav))/self.resolution))):
                a =  max(mat_wav) + self.resolution*(i+1)
                material_full = np.append(material_full, a)
                nr_material_full = np.append(nr_material_full, mat_real[max_index-1]) 
                k_material_full = np.append(k_material_full, mat_comp[max_index-1])

            mat_wav = np.append(mat_wav, material_full)
            mat_real = np.append(mat_real, nr_material_full)
            mat_comp = np.append(mat_comp, k_material_full)

        nr_material_full = np.array([])
        k_material_full = np.array([])    
        if self.min_wave < min(mat_wav):
            additional_waves = np.arange(self.min_wave,min(mat_wav),self.resolution)
            nr_material_full = np.append([mat_real[0] for i in range(np.shape(additional_waves)[0])],nr_material_full)
            k_material_full = np.append([mat_comp[0] for i in range(np.shape(additional_waves)[0])],k_material_full)

            mat_wav = np.append(additional_waves,mat_wav)
            mat_real = np.append(nr_material_full,mat_real)
            mat_comp = np.append(k_material_full,mat_comp)


        intermediate_n_material = interpolate.splrep(mat_wav, mat_real)
        nr_material = interpolate.splev(wave_spacing, intermediate_n_material)

        intermediate_k_material = interpolate.splrep(mat_wav, mat_comp)
        k_material = interpolate.splev(wave_spacing, intermediate_k_material)

        return nr_material, k_material

    #helper function    
    def search_from_nr_k_generator(self,arr,material):
        for element in arr:
            if element[0] == material:
                return element[1],element[2]

    #for example search_from_usp("Air",unique_set_props) would give properties of air
    #res = search_from_usp("Air",unique_set_props)
    #res[1] wavelengths
    #res[2] real parts
    #res[3] complex parts
    
    def s_pol(self,index):
        import numpy as np
    
    # consider the thickness of the layers

        layer_thickness = self.prescription[:, 1]
        #if user enters optical thickness file, convert back

    # creation of empty arrays and matrices

        nr = []
        k = []
        reflectance_s = []
        transmittance_s = []
        absorbance_s = []
        reflected_phase_shift_s = []
        transmitted_phase_shift_s = []
        angolini = []
        nr_previous_material = []
        k_previous_material = []
        final_s = np.identity(2)
        temp = np.eye(2)

        for l in range(self.number_of_layers):


            nr, k =   self.search_from_nr_k_generator(self.nr_k_array, self.prescription[l,0])

            N =  nr[index] - 1j*k[index]


            nr_previous_material = np.append(nr_previous_material, nr[index])
            k_previous_material = np.append(k_previous_material, k[index])

    # separate the case in which the previous material is the 'immersion material' or a layer in the stack

            if l == 0:
                M = self.N_medium
                starting_angle = self.starting_angle_2
                angle = np.arcsin((M/N)*np.sin(starting_angle))

            else:
                M = nr_previous_material[l-1] - 1j*k_previous_material[l-1]
                starting_angle = angle
                angle = np.arcsin((M/N)*np.sin(starting_angle))


            angolini = np.append(angolini,angle)

            eta_s = self.admitt*(np.sqrt(nr[index]**2 - k[index]**2 - self.nr_medium[index]**2*(np.sin(self.starting_angle_2))**2 - 2*1j*nr[index]*k[index]))
            delta = ((2*np.pi)/self.wave_spacing[index])*float(layer_thickness[l])*(np.sqrt(nr[index]**2 - k[index]**2 - self.nr_medium[index]**2*(np.sin(self.starting_angle_2))**2 - 2*1j*nr[index]*k[index]))

    # elements of the matrices

            topleft_s = np.cos(delta)
            topright_s = 1j*np.sin(delta)/eta_s
            bottomleft_s = 1j*(eta_s)*np.sin(delta)
            bottomright_s = np.cos(delta)

    # assemble the elements in a matrix        

            matrix_s = np.array([[topleft_s,topright_s],[bottomleft_s,bottomright_s]])

            temp = temp@matrix_s

            final_s = temp

    # take the real and imaginary refractive index parts for the substrate 

        nr_substrate, k_substrate =   self.search_from_nr_k_generator(self.nr_k_array,self.prescription[-1,0]) 

    # take the last layer-substrate incident angle

        angle = np.arcsin((N/(nr_substrate[index]- 1j*k_substrate[index]))*np.sin(angolini[-1]))

        substrate_s = np.array([[1],[(nr_substrate[index]- 1j*k_substrate[index])*np.cos(angle)*self.admitt]])

        final_s = final_s@substrate_s

    # take the B and C coefficients from the result

        B_s = final_s[0]

        C_s = final_s[1]

        Bmod_s = self.admitt*self.N_medium[index]*B_s*np.cos(self.starting_angle_2)

    # calculate the reflectance and transmittance    

        refl_s = ((Bmod_s - C_s)/(Bmod_s + C_s))*np.conjugate(((Bmod_s - C_s)/(Bmod_s + C_s)))

        trans_s = (4*self.admitt*self.N_medium[index]*np.cos(self.starting_angle_2)*((nr_substrate[index]- 1j*k_substrate[index])*np.cos(angle)).real*self.admitt)/((Bmod_s + C_s)*np.conjugate(Bmod_s + C_s))

        abs_s = (4*self.admitt*self.N_medium[index]*np.cos(self.starting_angle_2)*(B_s*np.conjugate(C_s)-(nr_substrate[index]- 1j*k_substrate[index])*self.admitt*np.cos(angle)).real)/((Bmod_s + C_s)*np.conjugate(Bmod_s + C_s))

        reflectance_s = np.append(reflectance_s, refl_s)

        transmittance_s = np.append(transmittance_s, trans_s)

        absorbance_s = np.append(absorbance_s, abs_s)

    # calculate the reflected phase shift    

        refl_ps_numerator =  (((nr_substrate[index]- 1j*k_substrate[index])*np.cos(angle)*self.admitt)*(B_s*np.conjugate(C_s)-(C_s*np.conjugate(B_s)))).imag 
        refl_ps_denominator = ((nr_substrate[index]- 1j*k_substrate[index])*np.cos(angle)*self.admitt)**2*B_s*np.conjugate(B_s)- C_s*np.conjugate(C_s)
        reflected_ps = np.arctan2(refl_ps_numerator.real, refl_ps_denominator.real)
        reflected_phase_shift_s = np.append(reflected_phase_shift_s, reflected_ps)

    # calculate the transmitted phase shift

        trans_ps_numerator = -(Bmod_s + C_s).imag
        trans_ps_denominator = (Bmod_s + C_s).real
        transmitted_ps = np.arctan2(trans_ps_numerator.real, trans_ps_denominator.real)
        transmitted_phase_shift_s = np.append(transmitted_phase_shift_s, transmitted_ps)

        return self.wave_spacing[index], transmittance_s, reflectance_s, absorbance_s, reflected_phase_shift_s, transmitted_phase_shift_s  
    
    def s_polarization(self,plot_label = 'transmittance',savefile = True,savefig = True):
        
        import numpy as np
        import matplotlib.pyplot as plt
        
        wave_spacing_index = [] #A
        transmittance = [] #B
        reflectance = [] #C
        absorbance = [] #D
        reflected_phase_shift = [] #E
        transmitted_phase_shift = [] #F

        aid= self.angle_in_deg 
        #TiO2_150nm = np.loadtxt(Path("phase_good_noheader.txt"))

        for w in range(np.size(self.wave_spacing)):
            
            a, b, c, d, rps, tps = self.s_pol(w)
            wave_spacing_index.append(a)
            transmittance.append(100*b.real)
            reflectance.append(100*c.real)
            absorbance.append(100*(d).real)
            reflected_phase_shift.append(np.rad2deg(rps.real))
            transmitted_phase_shift.append(np.rad2deg(tps.real))
        
        if self.refinement_type == None:  
            plt.figure(dpi=600)
            if plot_label == 'transmittance':
                plt.plot(wave_spacing_index,transmittance, color = 'red', label = 'My Algo')
                plt.title(f'Transmittance {self.angle_in_deg}° s-pol')
                plt.ylabel('Transmittance(%)')
                plt.xlabel('Wavelength(nm)')
                result_array = np.array([np.array(wave_spacing_index).ravel(),np.array(transmittance).ravel()]).T
            if plot_label == 'reflectance':
                plt.plot(wave_spacing_index,reflectance, color = 'red', label = 'My Algo')
                plt.title(f'Reflectance {self.angle_in_deg}° s-pol')
                plt.ylabel('Reflectance(%)')
                plt.xlabel('Wavelength(nm)')
                result_array = np.array([np.array(wave_spacing_index).ravel(),np.array(reflectance).ravel()]).T
            if plot_label == 'absorbance':
                plt.plot(wave_spacing_index,absorbance, color = 'red', label = 'My Algo')
                plt.title(f'Absorbance {self.angle_in_deg}° s-pol')
                plt.ylabel('Absorptance(%)')
                plt.xlabel('Wavelength(nm)')
                result_array = np.array([np.array(wave_spacing_index).ravel(),np.array(absorbance).ravel()]).T
            if plot_label == 'reflected_phase_shift':
                plt.plot(wave_spacing_index,reflected_phase_shift, color = 'red', label = 'My Algo')
                plt.title(f'Reflected phase shift {self.angle_in_deg}° s-pol')
                plt.ylabel('Reflected phase shift(°)')
                plt.xlabel('Wavelength(nm)')
                result_array = np.array([np.array(wave_spacing_index).ravel(),np.array(reflected_phase_shift).ravel()]).T
            if plot_label == 'transmitted_phase_shift':
                plt.plot(wave_spacing_index,transmitted_phase_shift, color = 'red', label = 'My Algo')
                plt.title(f'Transmitted phase shift {self.angle_in_deg}° s-pol')
                plt.ylabel('Transmitted phase shift(°)')
                plt.xlabel('Wavelength(nm)')
                result_array = np.array([np.array(wave_spacing_index).ravel(),np.array(transmitted_phase_shift).ravel()]).T
            if plot_label == 'GD_ref':
                phi_ = np.array(reflected_phase_shift)
                phi_ = np.ravel(phi_)
                lambda_  = np.array(wave_spacing_index)
                dphi_dlambda = np.gradient(phi_,lambda_)
                gd = 10**(-2)/(6*np.pi)*lambda_*lambda_*dphi_dlambda/57.296
                plt.plot(wave_spacing_index,gd, color = 'red', label = 'My Algo') 
                plt.title(f'GD[fs] {self.angle_in_deg}° s-pol')
                plt.ylim(-30, 30)
                plt.ylabel('Reflected GD(fs)')
                plt.xlabel('Wavelength(nm)')
                result_array = np.array([np.array(wave_spacing_index).ravel(),np.array(gd).ravel()]).T
            if plot_label == 'GDD_ref':
                phi_ = np.array(reflected_phase_shift)
                phi_ = np.ravel(phi_)
                lambda_  = np.array(wave_spacing_index)
                dphi_dlambda = np.gradient(phi_,lambda_)
                dphi_dlambda_2 = np.gradient(dphi_dlambda,lambda_)
                gdd = dphi_dlambda_2
                gdd = 10**(-4)/(36*np.pi**2)*(lambda_**4)*dphi_dlambda_2/57.296
                plt.plot(lambda_,gdd, color = 'red', label = 'My Algo') 
                plt.title(f'GDD[fs^2] {self.angle_in_deg}° s-pol')
                plt.ylim(-30, 30)
                plt.ylabel('Reflected GDD(fs²)')
                plt.xlabel('Wavelength(nm)')
                result_array = np.array([np.array(wave_spacing_index).ravel(),np.array(gdd).ravel()]).T
            if plot_label == 'GD_trans':
                phi_ = np.array(transmitted_phase_shift)
                phi_ = np.ravel(phi_)
                lambda_  = np.array(wave_spacing_index)
                dphi_dlambda = np.gradient(phi_,lambda_)
                gd = 10**(-2)/(6*np.pi)*lambda_*lambda_*dphi_dlambda/57.296
                plt.plot(wave_spacing_index,gd, color = 'red', label = 'My Algo') 
                plt.title(f'GD[fs] {self.angle_in_deg}° s-pol')
                plt.ylim(-30, 30)
                plt.ylabel('Transmitted GD(fs)')
                plt.xlabel('Wavelength(nm)')
                result_array = np.array([np.array(wave_spacing_index).ravel(),np.array(gd).ravel()]).T
            if plot_label == 'GDD_trans':
                phi_ = np.array(transmitted_phase_shift)
                phi_ = np.ravel(phi_)
                lambda_  = np.array(wave_spacing_index)
                dphi_dlambda = np.gradient(phi_,lambda_)
                dphi_dlambda_2 = np.gradient(dphi_dlambda,lambda_)
                gdd = dphi_dlambda_2
                gdd = 10**(-4)/(36*np.pi**2)*(lambda_**4)*dphi_dlambda_2/57.296
                plt.plot(lambda_,gdd, color = 'red', label = 'My Algo') 
                plt.title(f'GDD[fs^2] {self.angle_in_deg}° s-pol')
                plt.ylim(-30, 30)
                plt.ylabel('Transmitted GDD(fs²)')
                plt.xlabel('Wavelength(nm)')
                result_array = np.array([np.array(wave_spacing_index).ravel(),np.array(gdd).ravel()]).T
            if savefile == True:
                np.savetxt(plot_label+'.txt',result_array)
            if savefig == True:
                plt.savefig(plot_label)
            plt.legend()
        
        #test.getGD(wave_spacing_index,reflected_phase_shift)
        
        if plot_label == 'reflectance':
            return wave_spacing_index, reflectance
        if plot_label == 'transmittance':
            return wave_spacing_index, transmittance  
        if plot_label == 'absorbance':
            return wave_spacing_index, absorbance

        
  


    def p_pol(self,index):
        
        import numpy as np

        nr = []
        k = []
        layer_thickness = self.prescription[:, 1]
        reflectance_p = []
        transmittance_p = []
        absorbance_p = []
        reflected_phase_shift_p = []
        transmitted_phase_shift_p = []
        angolini = []
        nr_previous_material = []
        k_previous_material = []
        final_p = np.identity(2)
        temp = np.eye(2)

        for l in range(self.number_of_layers):

            nr, k =   self.search_from_nr_k_generator(self.nr_k_array,self.prescription[l,0])

            N =  nr[index] - 1j*k[index]

            nr_previous_material = np.append(nr_previous_material, nr[index])
            k_previous_material = np.append(k_previous_material, k[index])


            if l == 0:
                M = self.N_medium
                starting_angle = self.starting_angle_2
                angle = np.arcsin((M/N)*np.sin(starting_angle))

            else:
                M = nr_previous_material[l-1] - 1j*k_previous_material[l-1]
                starting_angle = angle
                angle = np.arcsin((M/N)*np.sin(starting_angle))



            angolini = np.append(angolini,angle)


            eta_s = self.admitt*(np.sqrt(nr[index]**2 - k[index]**2 - self.nr_medium[index]**2*(np.sin(self.starting_angle_2))**2 - 2*1j*nr[index]*k[index]))
            eta_p = (self.admitt**2*((nr[index] - 1j*k[index])**2))/eta_s
            delta = ((2*np.pi)/self.wave_spacing[index])*float(layer_thickness[l])*(np.sqrt(nr[index]**2 - k[index]**2 - self.nr_medium[index]**2*(np.sin(self.starting_angle_2))**2 - 2*1j*nr[index]*k[index]))

            topleft_p = np.cos(delta)
            topright_p = 1j*np.sin(delta)/eta_p
            bottomleft_p = 1j*(eta_p)*np.sin(delta)
            bottomright_p = np.cos(delta)

            matrix_p = np.array([[topleft_p,topright_p],[bottomleft_p,bottomright_p]])

            temp = temp@matrix_p

            final_p = temp


        nr_substrate, k_substrate =   self.search_from_nr_k_generator(self.nr_k_array,self.prescription[-1,0]) 

        angle = np.arcsin((N/(nr_substrate[index]- 1j*k_substrate[index]))*np.sin(angolini[-1]))

        substrate_p = np.array([[1],[(nr_substrate[index]- 1j*k_substrate[index])/np.cos(angle)*self.admitt]])
        final_p = final_p@substrate_p

        B_p = final_p[0]
        C_p = final_p[1]
        Bmod_p = self.admitt*self.N_medium[index]*B_p/np.cos(self.starting_angle_2)

        refl_p = ((Bmod_p - C_p)/(Bmod_p + C_p))*np.conjugate(((Bmod_p - C_p)/(Bmod_p + C_p)))

        trans_p = (4*self.admitt*self.N_medium[index]/np.cos(self.starting_angle_2)*((nr_substrate[index]- 1j*k_substrate[index])/np.cos(angle)).real*self.admitt)/((Bmod_p + C_p)*np.conjugate(Bmod_p + C_p))

        abs_p = (4*self.admitt*self.N_medium[index]/np.cos(self.starting_angle_2)*(B_p*np.conjugate(C_p)-(nr_substrate[index]- 1j*k_substrate[index])*self.admitt/np.cos(angle)).real)/((Bmod_p + C_p)*np.conjugate(Bmod_p + C_p))

        transmittance_p = np.append(transmittance_p, trans_p)
        reflectance_p = np.append(reflectance_p, refl_p)
        absorbance_p = np.append(absorbance_p, abs_p)

        refl_ps_numerator =  (((nr_substrate[index]- 1j*k_substrate[index])/np.cos(angle)*self.admitt)*(B_p*np.conjugate(C_p)-(C_p*np.conjugate(B_p)))).imag 
        refl_ps_denominator = ((nr_substrate[index]- 1j*k_substrate[index])/np.cos(angle)*self.admitt)**2*B_p*np.conjugate(B_p)- C_p*np.conjugate(C_p)
        reflected_ps = np.arctan2(refl_ps_numerator.real, refl_ps_denominator.real)
        reflected_phase_shift_p = np.append(reflected_phase_shift_p, reflected_ps)

        trans_ps_numerator = -(Bmod_p + C_p).imag
        trans_ps_denominator = (Bmod_p + C_p).real
        transmitted_ps = np.arctan2(trans_ps_numerator.real, trans_ps_denominator.real)
        transmitted_phase_shift_p = np.append(transmitted_phase_shift_p, transmitted_ps)
        
        

        return self.wave_spacing[index], transmittance_p, reflectance_p, absorbance_p, reflected_phase_shift_p, transmitted_phase_shift_p  
    
    
    def p_polarization(self,plot_label = 'transmittance',savefile = True,savefig = True):
        
        import numpy as np
        import matplotlib.pyplot as plt
        
        wave_spacing_index = [] #A
        transmittance = [] #B
        reflectance = [] #C
        absorbance = [] #D
        reflected_phase_shift = [] #E
        transmitted_phase_shift = [] #F

        aid= self.angle_in_deg 
        #TiO2_150nm = np.loadtxt(Path("phase_good_noheader.txt"))

        for w in range(np.size(self.wave_spacing)):
            
            a, b, c, d, rps, tps = self.p_pol(w)
            wave_spacing_index.append(a)
            transmittance.append(100*b.real)
            reflectance.append(100*c.real)
            absorbance.append(100*d.real)
            reflected_phase_shift.append(np.rad2deg(rps.real))
            transmitted_phase_shift.append(np.rad2deg(tps.real))
        if self.refinement_type == None:   
            plt.figure(dpi=600)
            if plot_label == 'transmittance':
                plt.plot(wave_spacing_index,transmittance, color = 'red', label = 'My Algo')
                plt.title(f'Transmittance {self.angle_in_deg}° p-pol')
                plt.ylabel('Transmittance(%)')
                plt.xlabel('Wavelength(nm)')
                result_array = np.array([np.array(wave_spacing_index).ravel(),np.array(transmittance).ravel()]).T
            if plot_label == 'reflectance':
                plt.plot(wave_spacing_index,reflectance, color = 'red', label = 'My Algo')
                plt.title(f'Reflectance {self.angle_in_deg}° p-pol')
                plt.ylabel('Reflectance(%)')
                plt.xlabel('Wavelength(nm)')
                result_array = np.array([np.array(wave_spacing_index).ravel(),np.array(reflectance).ravel()]).T
            if plot_label == 'absorbance':
                plt.plot(wave_spacing_index,absorbance, color = 'red', label = 'My Algo')
                plt.title(f'Absorbance {self.angle_in_deg}° p-pol')
                plt.ylabel('Absoptance(%)')
                plt.xlabel('Wavelength(nm)')
                result_array = np.array([np.array(wave_spacing_index).ravel(),np.array(absorbance).ravel()]).T
            if plot_label == 'reflected_phase_shift':
                plt.plot(wave_spacing_index,reflected_phase_shift, color = 'red', label = 'My Algo')
                plt.title(f'Reflected phase shift {self.angle_in_deg}° p-pol')
                plt.ylabel('Reflected phase shift(°)')
                plt.xlabel('Wavelength(nm)')
                result_array = np.array([np.array(wave_spacing_index).ravel(),np.array(reflected_phase_shift).ravel()]).T
            if plot_label == 'transmitted_phase_shift':
                plt.plot(wave_spacing_index,transmitted_phase_shift, color = 'red', label = 'My Algo')
                plt.title(f'Transmitted phase shift {self.angle_in_deg}° p-pol')
                plt.ylabel('Transmitted phase shift(°)')
                plt.xlabel('Wavelength(nm)')
                result_array = np.array([np.array(wave_spacing_index).ravel(),np.array(transmitted_phase_shift).ravel()]).T
            if plot_label == 'GD_ref':
                phi_ = np.array(reflected_phase_shift)
                phi_ = np.ravel(phi_)
                lambda_  = np.array(wave_spacing_index)
                dphi_dlambda = np.gradient(phi_,lambda_)
                gd = 10**(-2)/(6*np.pi)*lambda_*lambda_*dphi_dlambda/57.296
                plt.plot(wave_spacing_index,gd, color = 'red', label = 'My Algo') 
                plt.title(f'GD[fs] {self.angle_in_deg}° s-pol')
                plt.ylim(-30, 30)
                plt.ylabel('Reflected GD(fs)')
                plt.xlabel('Wavelength(nm)')
                result_array = np.array([np.array(wave_spacing_index).ravel(),np.array(gd).ravel()]).T
            if plot_label == 'GDD_ref':
                phi_ = np.array(reflected_phase_shift)
                phi_ = np.ravel(phi_)
                lambda_  = np.array(wave_spacing_index)
                dphi_dlambda = np.gradient(phi_,lambda_)
                dphi_dlambda_2 = np.gradient(dphi_dlambda,lambda_)
                gdd = dphi_dlambda_2
                gdd = 10**(-4)/(36*np.pi**2)*(lambda_**4)*dphi_dlambda_2/57.296
                plt.plot(lambda_,gdd, color = 'red', label = 'My Algo') 
                plt.title(f'GDD[fs²] {self.angle_in_deg}° s-pol')
                plt.ylabel('Reflected GDD(fs²)')
                plt.xlabel('Wavelength(nm)')
                plt.ylim(-30, 30)                
                result_array = np.array([np.array(wave_spacing_index).ravel(),np.array(gdd).ravel()]).T
            if plot_label == 'GD_trans':
                phi_ = np.array(transmitted_phase_shift)
                phi_ = np.ravel(phi_)
                lambda_  = np.array(wave_spacing_index)
                dphi_dlambda = np.gradient(phi_,lambda_)
                gd = 10**(-2)/(6*np.pi)*lambda_*lambda_*dphi_dlambda/57.296
                plt.plot(wave_spacing_index,gd, color = 'red', label = 'My Algo') 
                plt.title(f'GD[fs] {self.angle_in_deg}° s-pol')
                plt.ylim(-30, 30)
                plt.ylabel('Transmited GD(fs)')
                plt.xlabel('Wavelength(nm)')
                result_array = np.array([np.array(wave_spacing_index).ravel(),np.array(gd).ravel()]).T
            if plot_label == 'GDD_trans':
                phi_ = np.array(transmitted_phase_shift)
                phi_ = np.ravel(phi_)
                lambda_  = np.array(wave_spacing_index)
                dphi_dlambda = np.gradient(phi_,lambda_)
                dphi_dlambda_2 = np.gradient(dphi_dlambda,lambda_)
                gdd = dphi_dlambda_2
                gdd = 10**(-4)/(36*np.pi**2)*(lambda_**4)*dphi_dlambda_2/57.296
                plt.plot(lambda_,gdd, color = 'red', label = 'My Algo') 
                plt.title(f'GDD[fs^2] {self.angle_in_deg}° s-pol')
                plt.ylim(-30, 30)
                plt.ylabel('Transmited GDD(fs²)')
                plt.xlabel('Wavelength(nm)')
                result_array = np.array([np.array(wave_spacing_index).ravel(),np.array(gdd).ravel()]).T
            if savefile == True:
                np.savetxt(plot_label+'.txt',result_array)
            if savefig == True:
                plt.savefig(plot_label)
            plt.legend()
        
        if plot_label == 'reflectance':
            return wave_spacing_index, reflectance
        if plot_label == 'transmittance':
            return wave_spacing_index, transmittance
        if plot_label == 'absorbance':
            return wave_spacing_index, absorbance
 
    
    

    def admittance(self,savefile = True,savefig = True):
        
        import numpy as np
        import matplotlib.pyplot as plt
        
        layer_thickness = self.prescription[:, 1]
        wavelength = self.ref_wave
        FWOT = []
        Y_imag = []
        Y_real = []
        last_layer_opt_thickness = 0 
        nr_substrate, k_substrate =   self.search_from_nr_k_generator(self.nr_k_array,self.prescription[-1,0])
        nr_substrate_reference_wavelen = (nr_substrate)[self.wave_spacing == wavelength][0]
        substrate_s = np.array([[1],[nr_substrate_reference_wavelen]])
        temp = np.eye(2)
        
        for L in range(self.number_of_layers-1,0,-1):
            unity = float(layer_thickness[L])/1000
            if unity == 0:
                continue
            provina = np.arange(0, float(layer_thickness[L]) + unity, unity)
            nr, k =   self.search_from_nr_k_generator(self.nr_k_array,self.prescription[L,0])
            N = (nr)[self.wave_spacing == wavelength]
            Y = []
            FWOT.append(N[0]*provina/wavelength + last_layer_opt_thickness)
            last_layer_opt_thickness = FWOT[-1][-1]
            for i in provina:
                delta = 2*np.pi*N[0]*i/wavelength
                eta = N[0]
            # elements of the matrices
                topleft_s = np.cos(delta)
                topright_s = 1j*np.sin(delta)/eta
                bottomleft_s = 1j*(eta)*np.sin(delta)
                bottomright_s = np.cos(delta) 
            # assemble the elements in a matrix        
                matrix_s = np.array([[topleft_s,topright_s],[bottomleft_s,bottomright_s]])
                final_s = matrix_s@temp@substrate_s
            # take the B and C coefficients from the result
                B_s = final_s[0,0]
                C_s = final_s[1,0]
                Y = np.append(Y, C_s/B_s)
            temp = matrix_s@temp
            reale = Y.real
            imag = Y.imag
            Y_real.append(reale)
            Y_imag.append(imag)
        N_medium_ref = (self.nr_medium)[self.wave_spacing == wavelength]
        T_ref = 4*N_medium_ref*(substrate_s[1,0]).real/(((N_medium_ref*B_s+C_s)*((N_medium_ref*B_s+C_s).conjugate())).real)
        
        ad_real = []
        ad_imag = []
        fig,ax = plt.subplots(nrows = 1,ncols = 1)            
        for i in range(len(Y_real)):
            ad_real.extend(Y_real[i])
            ad_imag.extend(Y_imag[i])
            ax.plot(Y_real[i],Y_imag[i],label = 'layer_' + str(i+1))
        ax.axis('square')
        ax.hlines(y=0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linewidth=2, color='k')
        ax.set_xlabel('Re(Admittance)')
        ax.set_ylabel('Im(Admittance)')
        ax.legend(bbox_to_anchor=(1.1, 1.05))
        plt.show
        plt.tight_layout()
        if savefig == True:
            fig.savefig('admittance',dpi = 300)
        result_array = np.array([np.array(ad_real).ravel(),np.array(ad_imag).ravel()]).T
        if savefile == True:
            np.savetxt('admittance.txt',result_array)
        
    def electricField(self,savefile = True,savefig = True):
        
        import numpy as np
        import matplotlib.pyplot as plt
        layer_thickness = self.prescription[:, 1]
        wavelength = self.ref_wave
        FWOT = []
        Y_imag = []
        Y_real = []
        last_layer_opt_thickness = 0 
        nr_substrate, k_substrate =   self.search_from_nr_k_generator(self.nr_k_array,self.prescription[-1,0])
        nr_substrate_reference_wavelen = (nr_substrate)[self.wave_spacing == wavelength][0]
        substrate_s = np.array([[1],[nr_substrate_reference_wavelen]])
        temp = np.eye(2)
        
        for L in range(self.number_of_layers-1,0,-1):
            unity = float(layer_thickness[L])/1000
            if unity == 0:
                continue
            provina = np.arange(0, float(layer_thickness[L]) + unity, unity)
            nr, k =   self.search_from_nr_k_generator(self.nr_k_array,self.prescription[L,0])
            N = (nr)[self.wave_spacing == wavelength]
            Y = []
            FWOT.append(N[0]*provina/wavelength + last_layer_opt_thickness)
            last_layer_opt_thickness = FWOT[-1][-1]
            for i in provina:
                delta = 2*np.pi*N[0]*i/wavelength
                eta = N[0]
            # elements of the matrices
                topleft_s = np.cos(delta)
                topright_s = 1j*np.sin(delta)/eta
                bottomleft_s = 1j*(eta)*np.sin(delta)
                bottomright_s = np.cos(delta) 
            # assemble the elements in a matrix        
                matrix_s = np.array([[topleft_s,topright_s],[bottomleft_s,bottomright_s]])
                final_s = matrix_s@temp@substrate_s
            # take the B and C coefficients from the result
                B_s = final_s[0,0]
                C_s = final_s[1,0]
                Y = np.append(Y, C_s/B_s)
            temp = matrix_s@temp
            reale = Y.real
            imag = Y.imag
            Y_real.append(reale)
            Y_imag.append(imag)
        N_medium_ref = (self.nr_medium)[self.wave_spacing == wavelength]
        T_ref = 4*N_medium_ref*(substrate_s[1,0]).real/(((N_medium_ref*B_s+C_s)*((N_medium_ref*B_s+C_s).conjugate())).real)

        fig,ax = plt.subplots(nrows = 1,ncols = 1)
        FWOT_plot = []
        electricfield = []
        for i in range(len(Y_real)):
            FWOT_plot.extend(FWOT[-1][-1]-FWOT[i])
            electricfield.extend((2*T_ref/Y_real[i]/self.admitt)**0.5)
        ax.plot(FWOT_plot,electricfield,'tab:blue')
        ax.set_xlabel('Optical Distance (Full Wave Optical Thcikness) from Medium')
        ax.set_ylabel('Electric Field(V/m)')
        plt.show
        plt.tight_layout()
        if savefig == True:
            fig.savefig('electric_field',dpi = 300)
        result_array = np.array([np.array(FWOT_plot).ravel(),np.array(electricfield).ravel()]).T
        if savefile == True:
            np.savetxt('electricfield.txt',result_array)
        
    def s_polarizationcolor(self,label = 'transmittance',observer_type = 'CIE_1931',illuminant_type = 'A',color_space = 'XYZ'):
        
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import interpolate
        import cv2
        
        wave_spacing_index = [] #A
        transmittance = [] #B
        reflectance = [] #C
        absorbance = [] #D
        reflected_phase_shift = [] #E
        transmitted_phase_shift = [] #F

        aid= self.angle_in_deg
        #TiO2_150nm = np.loadtxt(Path("phase_good_noheader.txt"))

        for w in range(np.size(self.wave_spacing)):
           
            a, b, c, d, rps, tps = self.s_pol(w)
            wave_spacing_index.append(a)
            transmittance.append(100*b.real)
            reflectance.append(100*c.real)
            absorbance.append(d.real)
            reflected_phase_shift.append(np.rad2deg(rps.real))
            transmitted_phase_shift.append(np.rad2deg(tps.real))
       
       
        if observer_type == 'CIE_1931':
            all_data = np.loadtxt('cmf1931.txt')
            wave = all_data[40:, 0]
            x = all_data[40:, 1]
            y = all_data[40:, 2]
            z = all_data[40:, 3]

        if observer_type == 'CIE_1964':
            all_data = np.loadtxt('cmf1964.txt')
            wave = all_data[40:, 0]
            x = all_data[40:, 1]
            y = all_data[40:, 2]
            z = all_data[40:, 3]

        a = np.loadtxt('a.txt')
        wave_a = a[100:, 0]
        data_a = a[100:, 1]

        d65 = np.loadtxt('d65.txt')
        wave_d65 = d65[100:, 0]
        data_d65 = d65[100:, 1]

        e = np.loadtxt('e.txt')
        wave_e = e[100:, 0]
        data_e = e[100:, 1]

        resolution_color = 1
        min_wave_color = 400
        max_wave_color = 829 + self.resolution
        wave_spacing_color = np.arange(min_wave_color, max_wave_color, resolution_color)

        col_datum = []
        if label == 'transmittance':
            col_datum = transmittance
        if label == 'reflectance':
            col_datum = reflectance
           
           
        intermediate_reflectance = interpolate.splrep(wave_spacing_index, col_datum)
        reflectance_color = interpolate.splev(wave_spacing_color, intermediate_reflectance)

         #change to spol for s-polarisation #question
            

        if illuminant_type == 'A':
            limit = min(len(reflectance_color),len(data_a),len(x))
            #print(limit)
            numerator_X = np.trapz(data_a[:limit]*reflectance_color[:limit]*x[:limit])
            numerator_Y = np.trapz(data_a[:limit]*reflectance_color[:limit]*y[:limit])
            numerator_Z = np.trapz(data_a[:limit]*reflectance_color[:limit]*z[:limit])
            denominator = np.trapz(data_a[:limit]*y[:limit])

        if illuminant_type == 'E':
            limit = min(len(reflectance_color),len(data_e))
            numerator_X = np.trapz(data_e[:limit]*reflectance_color[:limit]*x[:limit])
            numerator_Y = np.trapz(data_e[:limit]*reflectance_color[:limit]*y[:limit])
            numerator_Z = np.trapz(data_e[:limit]*reflectance_color[:limit]*z[:limit])
            denominator = np.trapz(data_e[:limit]*y[:limit])

        if illuminant_type == 'D65':
            limit = min(len(reflectance_color),len(data_d65))
            numerator_X = np.trapz(data_d65[:limit]*reflectance_color[:limit]*x[:limit])
            numerator_Y = np.trapz(data_d65[:limit]*reflectance_color[:limit]*y[:limit])
            numerator_Z = np.trapz(data_d65[:limit]*reflectance_color[:limit]*z[:limit])
            denominator = np.trapz(data_d65[:limit]*y[:limit])

        X = numerator_X/denominator
        Y = numerator_Y/denominator
        Z = numerator_Z/denominator

        if color_space == 'XYZ':
            print("X,Y",X,Y)

        x = X/(X+Y+Z)
        y = Y/(X+Y+Z)

        if color_space == 'xyY':
            print("x,y",x,y)
        
        Xn = 0
        Yn = 0
        Zn = 0

        #illuminant_type = 'd65'

        if illuminant_type == 'd65':
            Xn = 95.0489
            Yn = 100
            Zn = 108.884

        if illuminant_type == 'd50':
            Xn = 96.4212
            Yn = 100
            Zn = 82.5188

        if illuminant_type == 'A':
            Xn = 100*(0.44757/0.40745)
            Yn = 100
            Zn = 100*(0.14498/0.40745)


        if illuminant_type == 'C':
            Xn = 100*(0.31006/0.31616)
            Yn = 100
            Zn = 100*(0.37378/0.31616)

        if illuminant_type == 'D55':
            Xn = 100*(0.3325/0.3476)
            Yn = 100
            Zn = 100*(0.3199/0.3476)

        if illuminant_type == 'E-equal-energy':
            Xn = 100*(0.333334/0.333330)
            Yn = 100
            Zn = 100*(0.333336/0.333330)


        def f(inp):
            if inp > 0.008856:
                return np.power(inp,1/3)
            else:
                return (16/116)+inp*7.787

        if Y/Yn > 0.008856:
            L = 116*f(Y/Yn) - 16
        else:
            L = 903.3 * Y/Yn
        a = 500*(f(X/Xn) - f(Y/Yn))
        b = 200*(f(Y/Yn) - f(Z/Zn))

        #print(L,a,b)
        if color_space =='LAB':
            print(L,a,b)

        #reference: https://www.sciencedirect.com/science/article/pii/B9780123919267500254#s0050

        illuminant_type = 'A'

        un_prime = 0
        vn_prime = 0

        if illuminant_type == 'd65':
            un_prime = 0.19793943
            vn_prime = 0.46831096

        if illuminant_type == 'd50':
            un_prime = 0.2091
            vn_prime = 0.4882

        if illuminant_type == 'A':
            un_prime = 0.2560
            vn_prime = 0.5243

        if illuminant_type == 'C':
            un_prime = 0.2009
            vn_prime = 0.4609

        if illuminant_type == 'D55':
            un_prime = 0.2044
            vn_prime = 0.4801

        if illuminant_type == 'E-equal-energy':
            un_prime = 0.2105
            vn_prime = 0.4737





        Y_n = 100

        u_prime = 4*x / ( -2*x + 12*y + 3 )
        v_prime = 9*y / ( -2*x + 12*y + 3 )



        def f(inp):
            if inp > 0.008856:
                return np.power(inp,1/3)
            else:
                return (16/116)+inp*7.787

        L = 0   

        if Y/Y_n > 0.008856:
            L = 116*f(Y/Y_n) - 16
        else:
            L = 903.3 * Y/Y_n
        U = 13*L*(u_prime-un_prime)
        V = 13*L*(v_prime-vn_prime)

        #print(L,U,V)
        if color_space == 'LUV':
            print(L,U,V)




        img = cv2.imread(r'CIE.png',1)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', img.shape[1], img.shape[0])

        img = cv2.circle(img, (int(365*x)+44, img.shape[0]- int(365*y)-35),radius=2, color=(0, 0, 255), thickness=-1)


        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




    def p_polarizationcolor(self,label = 'transmittance',observer_type = 'CIE_1931',illuminant_type = 'A',color_space = 'XYZ'):
        
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import interpolate
        import cv2
       
        wave_spacing_index = [] #A
        transmittance = [] #B
        reflectance = [] #C
        absorbance = [] #D
        reflected_phase_shift = [] #E
        transmitted_phase_shift = [] #F

        aid= self.angle_in_deg
        #TiO2_150nm = np.loadtxt(Path("phase_good_noheader.txt"))

        for w in range(np.size(self.wave_spacing)):
           
            a, b, c, d, rps, tps = self.p_pol(w)
            wave_spacing_index.append(a)
            transmittance.append(100*b.real)
            reflectance.append(100*c.real)
            absorbance.append(100*d.real)
            reflected_phase_shift.append(np.rad2deg(rps.real))
            transmitted_phase_shift.append(np.rad2deg(tps.real))
           
       

        #if color  == True:
            #observer_type = 'CIE_1931'

        if observer_type == 'CIE_1931':
            all_data = np.loadtxt('cmf1931.txt')
            wave = all_data[40:, 0]
            x = all_data[40:, 1]
            y = all_data[40:, 2]
            z = all_data[40:, 3]

        if observer_type == 'CIE_1964':
            all_data = np.loadtxt('cmf1964.txt')
            wave = all_data[40:, 0]
            x = all_data[40:, 1]
            y = all_data[40:, 2]
            z = all_data[40:, 3]

        a = np.loadtxt('a.txt')
        wave_a = a[100:, 0]
        data_a = a[100:, 1]

        d65 = np.loadtxt('d65.txt')
        wave_d65 = d65[100:, 0]
        data_d65 = d65[100:, 1]

        e = np.loadtxt('e.txt')
        wave_e = e[100:, 0]
        data_e = e[100:, 1]

        resolution_color = 1
        min_wave_color = 400
        max_wave_color = 829 + self.resolution
        wave_spacing_color = np.arange(min_wave_color, max_wave_color, resolution_color)

        col_datum = []
        if label == 'transmittance':
            col_datum = transmittance
        if label == 'reflectance':
            col_datum = reflectance
           
           
        intermediate_reflectance = interpolate.splrep(wave_spacing_index, col_datum)
        reflectance_color = interpolate.splev(wave_spacing_color, intermediate_reflectance)

        #illuminant_type = 'D65' #change to spol for s-polarisation #question

        if illuminant_type == 'A':
            limit = min(len(reflectance_color),len(data_a),len(x))
            #print(limit)
            numerator_X = np.trapz(data_a[:limit]*reflectance_color[:limit]*x[:limit])
            numerator_Y = np.trapz(data_a[:limit]*reflectance_color[:limit]*y[:limit])
            numerator_Z = np.trapz(data_a[:limit]*reflectance_color[:limit]*z[:limit])
            denominator = np.trapz(data_a[:limit]*y[:limit])

        if illuminant_type == 'E':
            limit = min(len(reflectance_color),len(data_e))
            numerator_X = np.trapz(data_e[:limit]*reflectance_color[:limit]*x[:limit])
            numerator_Y = np.trapz(data_e[:limit]*reflectance_color[:limit]*y[:limit])
            numerator_Z = np.trapz(data_e[:limit]*reflectance_color[:limit]*z[:limit])
            denominator = np.trapz(data_e[:limit]*y[:limit])

        if illuminant_type == 'D65':
            limit = min(len(reflectance_color),len(data_d65))
            numerator_X = np.trapz(data_d65[:limit]*reflectance_color[:limit]*x[:limit])
            numerator_Y = np.trapz(data_d65[:limit]*reflectance_color[:limit]*y[:limit])
            numerator_Z = np.trapz(data_d65[:limit]*reflectance_color[:limit]*z[:limit])
            denominator = np.trapz(data_d65[:limit]*y[:limit])

        X = numerator_X/denominator
        Y = numerator_Y/denominator
        Z = numerator_Z/denominator

        if color_space == 'XYZ':
            print("X,Y",X,Y)

        x = X/(X+Y+Z)
        y = Y/(X+Y+Z)

        if color_space == 'xyY':
            print("x,y",x,y)

        Xn = 0
        Yn = 0
        Zn = 0

        #illuminant_type = 'd65'

        if illuminant_type == 'd65':
            Xn = 95.0489
            Yn = 100
            Zn = 108.884

        if illuminant_type == 'd50':
            Xn = 96.4212
            Yn = 100
            Zn = 82.5188

        if illuminant_type == 'A':
            Xn = 100*(0.44757/0.40745)
            Yn = 100
            Zn = 100*(0.14498/0.40745)


        if illuminant_type == 'C':
            Xn = 100*(0.31006/0.31616)
            Yn = 100
            Zn = 100*(0.37378/0.31616)

        if illuminant_type == 'D55':
            Xn = 100*(0.3325/0.3476)
            Yn = 100
            Zn = 100*(0.3199/0.3476)

        if illuminant_type == 'E-equal-energy':
            Xn = 100*(0.333334/0.333330)
            Yn = 100
            Zn = 100*(0.333336/0.333330)


        def f(inp):
            if inp > 0.008856:
                return np.power(inp,1/3)
            else:
                return (16/116)+inp*7.787

        if Y/Yn > 0.008856:
            L = 116*f(Y/Yn) - 16
        else:
            L = 903.3 * Y/Yn
        a = 500*(f(X/Xn) - f(Y/Yn))
        b = 200*(f(Y/Yn) - f(Z/Zn))

        #print(L,a,b)
        if color_space == 'LAB':
            print(L,a,b)

        #reference: https://www.sciencedirect.com/science/article/pii/B9780123919267500254#s0050

        illuminant_type = 'A'

        un_prime = 0
        vn_prime = 0

        if illuminant_type == 'd65':
            un_prime = 0.19793943
            vn_prime = 0.46831096

        if illuminant_type == 'd50':
            un_prime = 0.2091
            vn_prime = 0.4882

        if illuminant_type == 'A':
            un_prime = 0.2560
            vn_prime = 0.5243

        if illuminant_type == 'C':
            un_prime = 0.2009
            vn_prime = 0.4609

        if illuminant_type == 'D55':
            un_prime = 0.2044
            vn_prime = 0.4801

        if illuminant_type == 'E-equal-energy':
            un_prime = 0.2105
            vn_prime = 0.4737





        Y_n = 100

        u_prime = 4*x / ( -2*x + 12*y + 3 )
        v_prime = 9*y / ( -2*x + 12*y + 3 )



        def f(inp):
            if inp > 0.008856:
                return np.power(inp,1/3)
            else:
                return (16/116)+inp*7.787

        L = 0   

        if Y/Y_n > 0.008856:
            L = 116*f(Y/Y_n) - 16
        else:
            L = 903.3 * Y/Y_n
        U = 13*L*(u_prime-un_prime)
        V = 13*L*(v_prime-vn_prime)

        #print(L,U,V)
        if color_space == 'LUV':
            print(L,U,V)



        img = cv2.imread(r'CIE.png',1)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', img.shape[1], img.shape[0])

        img = cv2.circle(img, (int(365*x)+44, img.shape[0]- int(365*y)-35),radius=2, color=(0, 0, 255), thickness=-1)


        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def refinement(self,tol = 10):
        
        import numpy as np
        from scipy.optimize import minimize
        from scipy.optimize import Bounds
        import matplotlib.pyplot as plt
        import time

        #refines transmittance or reflectance based on user input
        #the user input is in the form of a file target.txt
        #the file contains wavelength in the 1st column and the target % in the 2nd
        start_time = time.time()
        wl_target,f = np.loadtxt('target.txt')[:,0],np.loadtxt('target.txt')[:,1]
        wave_spacing_index = []
        R_s_initial = []
        R_p_initial = []
        T_s_initial = []
        T_p_initial = []
        for w in range(np.size(self.wave_spacing)):
            a, b, c, d, rps, tps = self.s_pol(w)
            wave_spacing_index.append(a)
            R_s_initial.append(100*c.real)
            T_s_initial.append(100*b.real)
            a, b, c, d, rps, tps = self.p_pol(w)
            R_p_initial.append(100*c.real)
            T_p_initial.append(100*b.real)
        R_mean_initial = 0.5*(np.array(R_p_initial) + np.array(R_s_initial))
        R_mean_initial = R_mean_initial[:,0]
        T_mean_initial = 0.5*(np.array(T_p_initial) + np.array(T_s_initial))
        T_mean_initial = T_mean_initial[:,0]
        wave_spacing_index = np.array(wave_spacing_index)
        
        '''***********defining function to be minimized*********************'''
        
        def return_difference(x):
            #returns the difference between user defined and calculated T or R
            self.prescription[:,1] = x
            if self.refinement_type == 'transmittance':
                wave_spacing_index = []
                T_s = []
                T_p = []
                for w in range(np.size(self.wave_spacing)):
                    a, b, c, d, rps, tps = self.s_pol(w)
                    wave_spacing_index.append(a)
                    T_s.append(100*b.real)
                    a, b, c, d, rps, tps = self.p_pol(w)
                    T_p.append(100*b.real)
                    
                T_mean = 0.5*(np.array(T_p) + np.array(T_s))
                T_mean = T_mean[:,0]
                wave_spacing_index = np.array(wave_spacing_index)
                diff = []
                for i in range(len(f)):
                    j = np.argmin(abs(wl_target[i] - wave_spacing_index))
                    diff.append(abs(f[i] - T_mean[j]))
                return np.linalg.norm(np.array(diff))/(len(f))**0.5
            elif self.refinement_type == 'reflectance':
                wave_spacing_index = []
                R_s = []
                R_p = []
                for w in range(np.size(self.wave_spacing)):
                    a, b, c, d, rps, tps = self.s_pol(w)
                    wave_spacing_index.append(a)
                    R_s.append(100*c.real)
                    a, b, c, d, rps, tps = self.p_pol(w)
                    R_p.append(100*c.real)
                R_mean = 0.5*(np.array(R_p) + np.array(R_s))
                R_mean = R_mean[:,0]
                wave_spacing_index = np.array(wave_spacing_index)
                diff = []
                for i in range(len(f)):
                    j = np.argmin(abs(wl_target[i] - wave_spacing_index))
                    diff.append(abs(f[i] - R_mean[j]))
                return np.linalg.norm(np.array(diff))/(len(f))**0.5
            
        '''*********defining initial condition and constraints**************'''
        x0 = (self.prescription)[:,1] 
        bounds = Bounds(0,np.inf)
        '''*******calling nelder mead simplex algorithm*********************'''
        res = minimize(return_difference, x0, method='nelder-mead',options={'xatol': tol, 'disp': True},bounds=bounds)
        print(res)
        '''*****************************************************************'''
        #converting results to optical thickness if needed
        layer_thickness = res.x
        if self.opt_thick == True:
            optical_thickness = []
            for l in range(self.number_of_layers):
                nr, k =   self.search_from_nr_k_generator(self.nr_k_array,self.prescription[l,0])
                N_ref = (nr)[self.wave_spacing == self.ref_wave][0]
                optical_thickness.append(float(layer_thickness[l])*N_ref/(self.ref_wave))
            optical_thickness = np.array(optical_thickness)
            refined_prescription = np.array([self.prescription[:,0],optical_thickness]).T
            np.savetxt('refined_prescription.txt',refined_prescription,fmt="%s",delimiter='\t')
        else:
            refined_prescription = np.array([self.prescription[:,0],layer_thickness]).T
            np.savetxt('refined_prescription.txt',refined_prescription,fmt="%s",delimiter='\t')
        end_time = time.time()
        print('Computational time (s) = ',end_time - start_time)
        
        R_s_refined = []
        R_p_refined = []
        T_s_refined = []
        T_p_refined = []
        for w in range(np.size(self.wave_spacing)):
            a, b, c, d, rps, tps = self.s_pol(w)
            R_s_refined.append(100*c.real)
            T_s_refined.append(100*b.real)
            a, b, c, d, rps, tps = self.p_pol(w)
            R_p_refined.append(100*c.real)
            T_p_refined.append(100*b.real)
        R_mean_refined = 0.5*(np.array(R_p_refined) + np.array(R_s_refined))
        R_mean_refined = R_mean_refined[:,0]
        T_mean_refined = 0.5*(np.array(T_p_refined) + np.array(T_s_refined))
        T_mean_refined = T_mean_refined[:,0]
        wave_spacing_index = np.array(wave_spacing_index)

        if self.refinement_type == 'reflectance':
            fig,ax = plt.subplots(nrows = 1,ncols = 1)
            ax.plot(wave_spacing_index,R_mean_initial, color = 'red', label = 'Initial')
            ax.plot(wl_target,f, color = 'blue', label = 'Target')
            ax.plot(wave_spacing_index,R_mean_refined, color = 'green', label = 'Refined')
            ax.set_title(f'Mean Reflectance {self.angle_in_deg}°')
            ax.set_ylabel('Reflectance(%)')
            ax.set_xlabel('Wavelength(nm)')
            ax.legend()
            fig.savefig('refinement_result',dpi = 300)
            initial = np.array([np.array(wave_spacing_index).ravel(),np.array(R_mean_initial).ravel()]).T
            final = np.array([np.array(wave_spacing_index).ravel(),np.array(R_mean_refined).ravel()]).T
            np.savetxt('Ref_before_refinement.txt',initial)
            np.savetxt('Ref_after_refinement.txt',final)
            
        if self.refinement_type == 'transmittance':
            fig,ax = plt.subplots(nrows = 1,ncols = 1)
            ax.plot(wave_spacing_index,T_mean_initial, color = 'red', label = 'Initial')
            ax.plot(wl_target,f, color = 'blue', label = 'Target')
            ax.plot(wave_spacing_index,T_mean_refined, color = 'green', label = 'Refined')
            ax.set_title(f'Mean Transmittance {self.angle_in_deg}°')
            ax.set_ylabel('Transmittance(%)')
            ax.set_xlabel('Wavelength(nm)')
            ax.legend()
            fig.savefig('refinement_result',dpi = 300)
            initial = np.array([np.array(wave_spacing_index).ravel(),np.array(T_mean_initial).ravel()]).T
            final = np.array([np.array(wave_spacing_index).ravel(),np.array(T_mean_refined).ravel()]).T
            np.savetxt('Trans_before_refinement.txt',initial)
            np.savetxt('Trans_after_refinement.txt',final)
